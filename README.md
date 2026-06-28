# FedSatGuard: Cross-Layer Anomaly Detection for Secure Federated Learning in Satellite-Terrestrial Networks

![JISA](https://www.sciencedirect.com/journal/journal-of-information-security-and-applications)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-orange)](https://pytorch.org/)

Official implementation of the paper:  
**"FedSatGuard: A Cross-Layer Anomaly Detection Framework for Secure and Trust-Adaptive Federated Learning in Satellite-Terrestrial Networks"**  
*The Journal of Information Security and Applications (JISA), 2026*

## 📌 Overview

FedSatGuard is a secure federated learning framework for satellite-terrestrial networks that fuses network telemetry with model update characteristics to detect cross-layer attacks. Key features:
- Cross-layer anomaly detection (Mahalanobis distance)
- Orbital-aware client scheduling
- Temporal trust modeling
- Blockchain-verified update admission
- Trust-adaptive aggregation

## 🚀 Key Results (from paper)

| Experiment | Result |
|------------|--------|
| Satellite Dynamics | 94% accuracy vs. FedAvg 40% |
| Real Starlink Data | 87–89% accuracy (2–4 point drop) |
| Poisoning Attack | +5% over Bulyan at 40% malicious |
| Byzantine Attack | +21% over Bulyan at 40% malicious |
| Adaptive Attack | 4.5 pt drop (vs. Bulyan 11.7 pt) |
| Tamper Resistance | 92–96% rejection rate |
| Overhead | 313 ms/round (faster than Krum/Bulyan) |

## 🔧 Requirements

- Python 3.8+
- PyTorch 1.9+
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn (for figures)

Install dependencies:
```bash
pip install -r requirements.txt
