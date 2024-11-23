from scipy.stats import levene
import pandas as pd

# Load datasets
DDoS_ICMP1_train = pd.read_csv('dataset/original/train_og/TCP_IP-DDoS-ICMP1_train.pcap.csv')
DDoS_ICMP1_test = pd.read_csv('dataset/original/test_og/TCP_IP-DDoS-ICMP1_test.pcap.csv')

# Prepare a report
f_test_results = {
    'column': [],
    'f-statistic': [],
    'p-value': []
}

# Perform F-test for each numerical column
for column in DDoS_ICMP1_train.columns:
    stat, p_value = levene(DDoS_ICMP1_train[column].dropna(), DDoS_ICMP1_test[column].dropna())  # Levene’s test
    f_test_results['column'].append(column)
    f_test_results['f-statistic'].append(stat)
    f_test_results['p-value'].append(p_value)

# Convert results to a DataFrame
f_test_results_df = pd.DataFrame(f_test_results)

for f in range(len(f_test_results_df)):
    if pd.notna(f_test_results_df.iloc[f, 2]) and f_test_results_df.iloc[f, 2] > 0.05:
        column_name = f_test_results_df.iloc[f,0] 
        p_value = f_test_results_df.iloc[f, 2] 
        print(f"{column_name}: p-value = {p_value:.2f}")

#---

DDoS_ICMP1_train = pd.read_csv('dataset/original/train_og/TCP_IP-DDoS-ICMP1_train.pcap.csv')
DDoS_ICMP2_test = pd.read_csv('dataset/original/test_og/TCP_IP-DDoS-ICMP2_test.pcap.csv')

# Prepare a report
f_test_results = {
    'column': [],
    'f-statistic': [],
    'p-value': []
}

# Perform F-test for each numerical column
for column in DDoS_ICMP1_train.columns:
    stat, p_value = levene(DDoS_ICMP1_train[column].dropna(), DDoS_ICMP2_test[column].dropna())  # Levene’s test
    f_test_results['column'].append(column)
    f_test_results['f-statistic'].append(stat)
    f_test_results['p-value'].append(p_value)

# Convert results to a DataFrame
f_test_results_df = pd.DataFrame(f_test_results)

for f in range(len(f_test_results_df)):
    if pd.notna(f_test_results_df.iloc[f, 2]) and f_test_results_df.iloc[f, 2] > 0.05:
        column_name = f_test_results_df.iloc[f,0] 
        p_value = f_test_results_df.iloc[f, 2] 
        print(f"{column_name}: p-value = {p_value:.2f}")
