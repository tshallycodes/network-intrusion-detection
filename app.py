import os
import re
import json
import pickle
import threading
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_auc_score
)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

MODEL_PATH = "model.pkl"
DATASET_DIR = os.path.expanduser("~/Documents/malware")
MAX_ROWS_PER_FILE = 50_000   # cap large files so training stays fast

model_data = {}
training_log = []
training_done = False
training_error = None
training_lock = threading.Lock()

# -----------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------

def normalise_label(val):
    """Return 'Benign' or 'Malicious' from any label variant."""
    if not isinstance(val, str):
        return "Unknown"
    v = val.strip()
    if v.lower().startswith("benign"):
        return "Benign"
    if "malicious" in v.lower():
        return "Malicious"
    return "Unknown"


def extract_attack_type(label_val, detail_val):
    """Return the attack sub-type string (C&C, DDoS, PortScan, …)."""
    combined = f"{label_val} {detail_val}".lower()
    if "portunhorizontal" in combined or "portscanhorizontal" in combined or "horizontalportscan" in combined:
        return "PortScan"
    if "portunhorizontal" in combined:
        return "PortScan"
    if "ddos" in combined:
        return "DDoS"
    if "c&c" in combined or "cc" in combined:
        return "C&C"
    if "filedownload" in combined:
        return "FileDownload"
    if "heartbeat" in combined:
        return "HeartBeat"
    if "attack" in combined:
        return "Attack"
    if "portscan" in combined:
        return "PortScan"
    return "Other"


def load_all_captures(dataset_dir):
    files = sorted([
        os.path.join(dataset_dir, f)
        for f in os.listdir(dataset_dir)
        if f.endswith(".csv")
    ])
    if not files:
        raise FileNotFoundError(f"No CSV files found in {dataset_dir}")

    dfs = []
    file_stats = []

    for fpath in files:
        fname = os.path.basename(fpath)
        try:
            df = pd.read_csv(fpath, sep="|", low_memory=False, dtype=str)
            df.columns = df.columns.str.strip()

            # drop header rows that snuck in as data
            if "label" in df.columns:
                df = df[df["label"] != "label"]

            # normalise label
            df["label_clean"] = df["label"].apply(normalise_label)
            df = df[df["label_clean"].isin(["Benign", "Malicious"])]

            # attack sub-type
            detail_col = "detailed-label" if "detailed-label" in df.columns else "detailed_label"
            df["attack_type"] = df.apply(
                lambda r: extract_attack_type(
                    r.get("label", ""),
                    r.get(detail_col, "")
                ), axis=1
            )

            n_total = len(df)
            n_benign = (df["label_clean"] == "Benign").sum()
            n_mal = (df["label_clean"] == "Malicious").sum()

            # sample large files proportionally
            if n_total > MAX_ROWS_PER_FILE:
                df = df.sample(n=MAX_ROWS_PER_FILE, random_state=42)

            file_stats.append({
                "file": fname,
                "total": n_total,
                "benign": int(n_benign),
                "malicious": int(n_mal),
                "sampled": len(df),
            })
            dfs.append(df)

        except Exception as e:
            file_stats.append({"file": fname, "error": str(e)})

    combined = pd.concat(dfs, ignore_index=True)
    return combined, file_stats


# -----------------------------------------------------------------
# Feature engineering
# -----------------------------------------------------------------

CATEGORICAL_COLS = [
    "proto", "conn_state", "history",
    "orig_bytes", "resp_bytes",
    "orig_pkts", "orig_ip_bytes",
    "resp_pkts", "resp_ip_bytes",
]

# id.resp_p is kept as numeric (port number)
NUMERIC_COLS = ["duration", "id.resp_p"]

DROP_COLS = [
    "ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h",
    "tunnel_parents", "label", "detailed-label", "detailed_label",
    "service", "missed_bytes", "local_orig", "local_resp",
    "label_clean", "attack_type",
]


def engineer_features(df):
    # numeric cols
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = df[col].replace("-", "0").fillna("0")
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # target
    le = LabelEncoder()
    df["label_encoded"] = le.fit_transform(df["label_clean"])

    # one-hot encode categoricals present in df
    ohe_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore", max_categories=30)
    encoded = ohe.fit_transform(df[ohe_cols].fillna("-"))
    ohe_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(ohe_cols), index=df.index)
    df = pd.concat([df, ohe_df], axis=1)

    # drop unneeded cols
    df.drop(columns=[c for c in DROP_COLS + ohe_cols if c in df.columns], inplace=True)

    # variance filter
    feature_cols = [c for c in df.columns if c != "label_encoded"]
    vt = VarianceThreshold(0.0)
    mask = vt.fit(df[feature_cols]).variances_ > 0
    valid = [f for f, v in zip(feature_cols, mask) if v]

    return df[valid + ["label_encoded"]], ohe, ohe_cols, le


def select_top_features(df, n=25):
    X = df.drop(columns=["label_encoded"])
    y = df["label_encoded"]
    rf_sel = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1, max_depth=8)
    rf_sel.fit(X, y)
    fi = pd.DataFrame({"Feature": X.columns, "Importance": rf_sel.feature_importances_})
    fi = fi.sort_values("Importance", ascending=False)
    top = fi.head(n)["Feature"].tolist()
    return top, fi


# -----------------------------------------------------------------
# Training
# -----------------------------------------------------------------

def train_model():
    global model_data, training_done, training_error, training_log

    with training_lock:
        training_log = []
        training_done = False
        training_error = None

    def log(msg):
        training_log.append(msg)

    try:
        log(f"Loading captures from {DATASET_DIR} ...")
        df, file_stats = load_all_captures(DATASET_DIR)
        log(f"Loaded {len(df):,} rows from {len(file_stats)} files")

        benign = (df["label_clean"] == "Benign").sum()
        malicious = (df["label_clean"] == "Malicious").sum()
        log(f"Benign: {benign:,}  |  Malicious: {malicious:,}")

        # attack type breakdown
        attack_breakdown = (
            df[df["label_clean"] == "Malicious"]["attack_type"]
            .value_counts().to_dict()
        )

        log("Engineering features ...")
        df_enc, ohe, ohe_cols, le = engineer_features(df)
        log(f"Feature matrix: {df_enc.shape[0]:,} rows × {df_enc.shape[1]-1} features")

        log("Selecting top 25 features ...")
        top_features, fi_df = select_top_features(df_enc, n=25)
        log(f"Top feature: {top_features[0]}")

        X = df_enc[top_features]
        y = df_enc["label_encoded"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        log(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

        log("Training Random Forest ...")
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred).tolist()
        report = classification_report(
            y_test, y_pred,
            target_names=["Benign", "Malicious"],
            output_dict=True
        )
        log(f"Accuracy: {acc:.4f}  |  ROC-AUC: {auc:.4f}")

        with training_lock:
            model_data = {
                "model": clf,
                "ohe": ohe,
                "ohe_cols": ohe_cols,
                "le": le,
                "top_features": top_features,
                "fi_df": fi_df.head(25).to_dict(orient="records"),
                "accuracy": round(acc * 100, 2),
                "auc": round(auc * 100, 2),
                "cm": cm,
                "report": report,
                "benign_count": int(benign),
                "malicious_count": int(malicious),
                "total": int(len(df)),
                "file_stats": file_stats,
                "attack_breakdown": attack_breakdown,
            }
            training_done = True
            log("Training complete.")

        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model_data, f)

    except Exception as e:
        with training_lock:
            training_error = str(e)
            training_done = True
        training_log.append(f"ERROR: {e}")
        import traceback
        training_log.append(traceback.format_exc())


# -----------------------------------------------------------------
# Routes
# -----------------------------------------------------------------

@app.route("/")
def index():
    dataset_exists = os.path.isdir(DATASET_DIR) and any(
        f.endswith(".csv") for f in os.listdir(DATASET_DIR)
    )
    loaded = bool(model_data)
    return render_template("index.html",
                           dataset_exists=dataset_exists,
                           loaded=loaded,
                           model_data=model_data if loaded else None)


@app.route("/train", methods=["POST"])
def train():
    global training_done
    if not os.path.isdir(DATASET_DIR):
        return jsonify({"error": f"Dataset directory not found: {DATASET_DIR}"}), 400
    training_done = False
    t = threading.Thread(target=train_model)
    t.daemon = True
    t.start()
    return jsonify({"status": "started"})


@app.route("/status")
def status():
    with training_lock:
        return jsonify({
            "done": training_done,
            "log": list(training_log),
            "error": training_error
        })


@app.route("/results")
def results():
    if not model_data:
        return jsonify({"error": "Model not trained yet"}), 400
    return jsonify({
        "accuracy": model_data["accuracy"],
        "auc": model_data["auc"],
        "cm": model_data["cm"],
        "report": model_data["report"],
        "fi": model_data["fi_df"],
        "benign_count": model_data["benign_count"],
        "malicious_count": model_data["malicious_count"],
        "total": model_data["total"],
        "file_stats": model_data.get("file_stats", []),
        "attack_breakdown": model_data.get("attack_breakdown", {}),
    })


@app.route("/predict", methods=["POST"])
def predict():
    if not model_data:
        return jsonify({"error": "Model not trained yet"}), 400

    data = request.get_json()
    clf = model_data["model"]
    ohe = model_data["ohe"]
    ohe_cols = model_data["ohe_cols"]
    top_features = model_data["top_features"]

    try:
        row = pd.DataFrame([data])

        # numeric
        for col in NUMERIC_COLS:
            if col in row.columns:
                row[col] = pd.to_numeric(row[col].replace("-", "0"), errors="coerce").fillna(0)

        # one-hot
        encoded = ohe.transform(row[ohe_cols].fillna("-"))
        ohe_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(ohe_cols))
        row = pd.concat([row.reset_index(drop=True), ohe_df.reset_index(drop=True)], axis=1)

        for m in [f for f in top_features if f not in row.columns]:
            row[m] = 0.0

        X_pred = row[top_features]
        pred = clf.predict(X_pred)[0]
        prob = clf.predict_proba(X_pred)[0]

        return jsonify({
            "label": "Malicious" if pred == 1 else "Benign",
            "benign_prob": round(float(prob[0]) * 100, 1),
            "malicious_prob": round(float(prob[1]) * 100, 1),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                model_data = pickle.load(f)
            training_done = True
            print("Loaded cached model.")
        except Exception:
            print("Cache invalid — retrain via the UI.")
    app.run(debug=True, port=5016)
