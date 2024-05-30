import os
import argparse
import subprocess
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import shutil

def normalize_and_save(experiment_type, window_size, offset, base_dir="C:\\Users\\willi\\CAN_experiments"):
    os.makedirs(base_dir, exist_ok=True)
    experiment_dir = os.path.join(base_dir, f"{experiment_type}_{window_size}_{offset}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    source_dir = "C:\\Users\\willi"
    csv_files = [f for f in os.listdir(source_dir) if f.endswith('.csv') and f.startswith(experiment_type)]
    
    for csv_file in csv_files:
        # Copy the original CSV file to the experiment directory
        shutil.copy(os.path.join(source_dir, csv_file), experiment_dir)
        
        # Read the copied CSV file
        df = pd.read_csv(os.path.join(experiment_dir, csv_file))
        
        # Normalize individual DataFrame
        scaler = MinMaxScaler()
        df[:] = scaler.fit_transform(df)
        
        # Fill missing values with zeros
        all_columns = set(df.columns)
        df = df.reindex(columns=all_columns, fill_value=0)

        # Save normalized individual DataFrame
        df.to_csv(os.path.join(experiment_dir, f"normalized_{csv_file}"), index=False)
    
    # Combine all normalized CSV files into a single DataFrame
    normalized_csv_files = [f for f in os.listdir(experiment_dir) if f.startswith('normalized_')]
    dataframes = [pd.read_csv(os.path.join(experiment_dir, file)) for file in normalized_csv_files]
    
    # Concatenate DataFrames along columns
    final_df = pd.concat(dataframes, axis=1).fillna(0)

    # Save the concatenated DataFrame
    final_df.to_csv(os.path.join(experiment_dir, f"{experiment_type}_all_w{window_size}_off{offset}_normalized.csv"), index=False)

def main():
    parser = argparse.ArgumentParser(description="Run experiments with specified window sizes and offsets.")
    parser.add_argument("--window-size", nargs='+', type=int, required=True, help="Window sizes to use, space-separated if multiple.")
    parser.add_argument("--offset", nargs='+', type=int, required=True, help="Offsets to use, space-separated if multiple.")
    args = parser.parse_args()

    process_all_files_script = "C:\\Users\\willi\\process_all_files.py"
    process_all_attacks_script = "C:\\Users\\willi\\process_all_correlated_masquerade_attack_files.py"

    for window_size in args.window_size:
        for offset in args.offset:
            subprocess.run(["python", process_all_files_script, "--window-size", str(window_size), "--offset", str(offset)])
            normalize_and_save("benign", window_size, offset)
            
            subprocess.run(["python", process_all_attacks_script, "--window-size", str(window_size), "--offset", str(offset)])
            normalize_and_save("attack", window_size, offset)
            
            print(f"Completed processing and normalization for window size {window_size} and offset {offset}")

if __name__ == "__main__":
    main()
