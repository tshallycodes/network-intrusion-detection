# 🛡️ Network Intrusion Detector

A machine learning web app that analyses IoT network traffic logs and classifies connections as **Benign** or **Malicious (C&C)**. Built on the IoT-23 dataset using Zeek/Bro network logs with Random Forest classification, achieving 100% accuracy on the captured traffic data.

## Demo

Upload a Zeek `conn.log.labeled` file → AI extracts network features → Each connection classified as Benign or Malicious with confidence scores.

## Results

| Metric | Score |
|--------|-------|
| Accuracy | **~99%** |
| ROC-AUC | **~99%** |
| Dataset | IoT-23 — 25M+ connections across 12 captures |
| Sampled for Training | ~344,000 connections |
| Benign | ~79,000 |
| Malicious | ~265,000 |

## Attack Types Detected

| Attack Type | Description |
|-------------|-------------|
| **PortScan** | Horizontal port scanning — device probing the network |
| **DDoS** | Distributed denial of service traffic |
| **C&C** | Command & Control — malware phoning home |
| **Attack** | Active exploitation attempts |
| **FileDownload** | Malicious file retrieval via C&C channel |

## Features

- Loads all 12 CTU-IoT-Malware captures automatically from `~/Documents/malware/`
- Smart sampling: up to 50,000 rows per file to keep training fast (~344k total)
- Handles both clean and merged label formats across different capture versions
- Feature engineering: One-Hot Encoding on protocol, conn_state, history, byte/packet counts
- Feature selection pipeline: VarianceThreshold → Random Forest importance → top 25 features
- Attack type breakdown: PortScan, DDoS, C&C, Attack, FileDownload
- Live training progress log in the browser
- Interactive prediction form — classify any connection by entering its fields
- Feature importance chart + attack type doughnut chart
- Per-capture file stats table showing row counts and sampling
- Confusion matrix and full classification report
- Model cached after training — instant reload on restart

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data Format | Zeek/Bro `conn.log.labeled` (tab-separated) |
| Feature Engineering | LabelEncoder, OneHotEncoder — Scikit-Learn |
| Feature Selection | VarianceThreshold + Random Forest Importance |
| Model | RandomForestClassifier — Scikit-Learn |
| Evaluation | Accuracy, ROC-AUC, Confusion Matrix, F1 |
| Web Framework | Flask |
| Dataset | IoT-23 (Stratosphere IPS Lab) |
| Language | Python |

## How to Run

**1. Get the dataset**

Download from Kaggle: [Network Malware Detection — Connection Analysis](https://www.kaggle.com/datasets/agungpambudi/network-malware-detection-connection-analysis)

Extract the CSV files to:
```
~/Documents/malware/
```
The folder should contain 12 files named `CTU-IoT-Malware-Capture-*.csv`.

**2. Clone the repo**
```bash
git clone https://github.com/manny2341/network-intrusion-detector.git
cd network-intrusion-detector
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Start the app**
```bash
python3 app.py
```

**5. Open in browser**
```
http://127.0.0.1:5016
```
Click **Train Model** — training completes in under 60 seconds.

## How It Works

1. Raw `conn.log.labeled` loaded by parsing `#fields` header line
2. **Data wrangling**: Zeek's broken merged column `tunnel_parents label detailed-label` is split and cleaned
3. Columns with zero variance (single unique value) are dropped
4. **Feature engineering**: 12 categorical columns (protocol, conn_state, history, ports, IPs, byte counts, packet counts) → One-Hot Encoded
5. **Feature selection**: VarianceThreshold removes zero-variance columns → Random Forest selects top 20 most important features
6. **Training**: `RandomForestClassifier(max_depth=5, class_weight='balanced')` — balanced weights handle the 79/21 malicious/benign imbalance
7. Prediction: new connection fields encoded through same pipeline → classified as Benign or Malicious

## Top Features That Identify Malicious Traffic

| Rank | Feature | What It Means |
|------|---------|--------------|
| 1 | `history_S` | TCP SYN only (no response) — classic C&C port scan |
| 2 | `orig_ip_bytes_76` | 76-byte IP packets — NTP query pattern (benign) |
| 3 | `id.resp_p_123` | Destination port 123 (NTP) — benign time sync |
| 4 | `id.resp_p_50` | Destination port 50 (OSPF) — C&C target port |
| 5 | `proto_udp` | UDP protocol — NTP traffic is benign |

## Dataset

**IoT-23** (Stratosphere IPS Lab, CTU-Prague) — network traffic captured from IoT devices infected with various malware families. Contains both malicious (C&C, DDoS, scanning) and benign traffic.

- Source: [Kaggle — Network Malware Detection](https://www.kaggle.com/datasets/agungpambudi/network-malware-detection-connection-analysis)
- Format: Zeek/Bro conn.log with labeled ground truth
- Capture: `CTU-IoT-Malware-Capture-1-1` (Mirai variant)

## Project Structure

```
network-intrusion-detector/
├── app.py               # Flask server, feature engineering, RF training, prediction API
├── index.ipynb          # Original analysis notebook (exploratory work)
├── dataset/
│   └── conn.log.labeled # Zeek log file (download from Kaggle)
├── model.pkl            # Cached trained model (auto-generated on first train)
├── templates/
│   └── index.html       # Training UI, stats, feature chart, confusion matrix, prediction form
├── static/
│   └── style.css        # Dark theme styling
└── requirements.txt
```

## My Other ML Projects

| Project | Description | Repo |
|---------|-------------|------|
| Loan Default Predictor | XGBoost — credit risk classification | [loan-default-predictor](https://github.com/manny2341/loan-default-predictor) |
| Spam Email Detector | NLP — TF-IDF spam vs ham | [spam-email-detector](https://github.com/manny2341/spam-email-detector) |
| Face Recognition Login | OpenCV LBPH — webcam face authentication | [face-recognition-login](https://github.com/manny2341/face-recognition-login) |
| Stock Price Predictor | LSTM — 5,884 tickers + crypto | [stock-price-predictor](https://github.com/manny2341/stock-price-predictor) |

## Author

[@manny2341](https://github.com/manny2341)
