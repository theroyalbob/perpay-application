import pandas as pd
import os

# Read the CSV file
input_file = 'data/indego-trips-2021-q4.csv'
print(f"Reading {input_file}...")

# Read the CSV file in chunks to handle large file size
chunk_size = 100000
chunks = []

for chunk in pd.read_csv(input_file, chunksize=chunk_size):
    # Convert start_time to datetime if it's not already
    if 'start_time' in chunk.columns:
        chunk['start_time'] = pd.to_datetime(chunk['start_time'])
        # Filter out September data (month 9)
        chunk = chunk[chunk['start_time'].dt.month != 9]
    chunks.append(chunk)
    print(f"Processed {len(chunks) * chunk_size} rows...")

# Combine all chunks
print("Combining chunks...")
df = pd.concat(chunks, ignore_index=True)

# Save the cleaned data
output_file = 'data/indego-trips-2021-q4-cleaned.csv'
print(f"Saving cleaned data to {output_file}...")
df.to_csv(output_file, index=False)

print("Done!") 