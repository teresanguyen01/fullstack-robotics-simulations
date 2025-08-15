"""
Script for fixing sensor data only.
"""

import os
import pandas as pd
import argparse

def fix_time_wraparound(time_series):
    corrected_time = [time_series.iloc[0]]
    for i in range(1, len(time_series)):
        delta = time_series.iloc[i] - time_series.iloc[i - 1]
        if delta < 0:
            delta += 65536
        corrected_time.append(corrected_time[-1] + delta)
    return pd.Series(corrected_time)

def process_sensor_files(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            filepath = os.path.join(input_dir, filename)
            print(f"Processing {filename}")

            df = pd.read_csv(filepath)

            if df.shape[0] < 2:
                print(f"Skipping {filename}: not enough rows")
                continue

            df = df.iloc[1:].reset_index(drop=True)

            if 'Time_ms' not in df.columns:
                print(f"Skipping {filename}: 'Time_ms' column missing")
                continue

            df['Time_ms'] = fix_time_wraparound(df['Time_ms'])
            df['Time_ms'] -= df['Time_ms'].iloc[0]  # Re-zero

            outpath = os.path.join(output_dir, filename)
            df.to_csv(outpath, index=False)
            print(f"Saved to {outpath}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Fix sensor Time_ms wraparound and drop first row")
    p.add_argument("--input_dir", help="Directory containing input sensor CSVs")
    p.add_argument("--output_dir", help="Directory to write cleaned CSVs")
    args = p.parse_args()
    
    process_sensor_files(args.input_dir, args.output_dir)