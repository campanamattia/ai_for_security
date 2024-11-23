import pandas as pd
import glob

# List all the CSV files
csv_files = glob.glob('dataset/original/train_og/*.csv')  # This will match all CSV files in the current directory

# Dictionary to store DataFrames
dataframes_train = {}

# Loop through each file and read it into a DataFrame
for file in csv_files:
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file)
        dataframes_train[file] = df  # Store the DataFrame in the dictionary with filename as key
        print(f"Loaded {file}")
    except Exception as e:
        print(f"Error loading {file}: {e}")

# Example: Accessing a DataFrame by filename
# dataframes['ARP_Spoofing_train.pcap.csv']

# Sort the dictionary keys (file names) alphabetically
sorted_files = sorted(dataframes_train.keys())

# Loop through the sorted file names to print row counts
for file in sorted_files:
    df = dataframes_train[file]
    print(f"{file} has {df.shape[0]} rows.")
    
# random same row numbers division? 

csv_files = glob.glob('dataset/original/test_og/*.csv')
dataframes_test = {}

for file in csv_files:
    try:
        df = pd.read_csv(file)
        dataframes_test[file] = df
        print(f"Loaded {file}")
    except Exception as e:
        print(f"Error loading {file}: {e}")

sorted_files = sorted(dataframes_test.keys())

for file in sorted_files:
    df = dataframes_test[file]
    print(f"{file} has {df.shape[0]} rows.")

dataframes_test['dataset/original/test_og/TCP_IP-DoS-UDP_test.pcap.csv'].info()