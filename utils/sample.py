import pandas as pd
import random

# Set the seed for reproducibility
random.seed(42)

# Read the original CSV file
input_csv = 'dataset/train_binary.csv'
df = pd.read_csv(input_csv)

# Take a random sample of the data
sample_size = 100  # Adjust the sample size as needed
df_sample = df.sample(n=sample_size)

# Write the sample to a new CSV file
output_csv = 'dataset/output_sample.csv'
df_sample.to_csv(output_csv, index=False)

print(f"Sampled data saved to {output_csv}")