# IoMT Network Security Analysis Project

## Table of Contents
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Methodology](#methodology)
4. [Results](#model-performance)
5. [Technologies Used](#technologies-used)
6. [Conclusions](#conclusions)

## Overview
This project analyzes the CICIoMT2024 dataset to develop and evaluate machine learning models for detecting cyber threats in Internet of Medical Things (IoMT) networks. Our analysis focuses on identifying various types of attacks while evaluating different machine learning approaches for their effectiveness in securing medical device networks.

## Dataset
The CICIoMT2024 dataset contains network traffic from:
- 40 IoMT devices (25 real and 15 simulated)
- Multiple protocols: Wi-Fi, MQTT, and Bluetooth
- 18 different types of attacks

### Attack Categories
| Category | Description | Examples |
|----------|-------------|-----------|
| DDoS | Distributed Denial of Service | TCP, UDP, ICMP floods |
| DoS | Denial of Service | TCP, UDP, ICMP, SYN attacks |
| Reconnaissance | Network scanning | Port scan, OS scan, Vulnerability scan |
| MQTT | Protocol-specific attacks | Connect flood, Publish flood |
| Spoofing | Identity forgery | ARP Spoofing |

## Methodology

### Data Processing Pipeline
1. Dataset Preparation
   - Merged multiple CSV files
   - Created additional features for attack classification
   - Implemented balanced sampling
   
2. Feature Selection Methods
   - Manual Selection based on correlation analysis
   - Hierarchical Clustering with complete-linkage

## Model Performance

### Binary Classification Results
| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|---------|-----------|
| Logistic Regression | 91.6% | 92.0% | 91.6% | 91.6% |
| Random Forest | 99.7% | 99.7% | 99.7% | 99.7% |
| XGBoost | 99.7% | 99.7% | 99.7% | 99.7% |

### Multiclass Classification Results
| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|---------|-----------|
| Logistic Regression | 75.7% | 79.1% | 69.2% | 70.6% |
| Random Forest | 99.6% | 98.7% | 97.1% | 97.8% |
| XGBoost | 99.5% | 98.3% | 97.1% | 97.7% |
| Neural Network (6 classes) | 80.4% | 82.1% | 78.9% | 79.5% |
| Neural Network (5 classes) | 97.6% | 96.8% | 95.9% | 96.3% |

## Technologies Used

### Core Libraries
- [Python](https://www.python.org/doc/) - Primary programming language
- [NumPy](https://numpy.org/doc/) - Numerical computing
- [Pandas](https://pandas.pydata.org/docs/) - Data manipulation and analysis

### Machine Learning
- [Scikit-learn](https://scikit-learn.org/stable/documentation.html) - Traditional ML algorithms
- [PyTorch](https://pytorch.org/docs/stable/index.html) - Neural network implementation
- [XGBoost](https://xgboost.readthedocs.io/) - Gradient boosting

### Visualization
- [Matplotlib](https://matplotlib.org/stable/contents.html) - Basic plotting
- [Seaborn](https://seaborn.pydata.org/) - Statistical data visualization

## Conclusions

### Key Findings
1. **Supervised Learning Excellence**
   - Tree-based ensemble methods consistently achieved >99% accuracy
   - Merging DoS/DDoS categories improved neural network performance

2. **Unsupervised Learning Limitations**
   - Clustering algorithms struggled to naturally group attack types
   - Traditional anomaly detection proved less effective due to feature distributions

### Recommendations
1. For real-time threat detection in IoMT networks, implement ensemble methods (Random Forest or XGBoost)
2. Consider attack similarity when designing classification systems
3. Focus on supervised learning approaches over unsupervised methods for this specific use case

---
For more detailed information about specific implementations or to contribute to this project, please check our documentation or open an issue.
