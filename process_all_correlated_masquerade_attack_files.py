#process_all_correlated_masquerade_attack_files.py

import subprocess
import os

def main():
    # The directory where the new log files are located
    log_directory = "C:\\Users\\willi\\OneDrive\\Desktop\\Research\\oak_ridge_in_vehicle\\road\\attacks\\"

    # Mapping of log files to their corresponding .pkl folder names
    log_to_pkl = {
        "correlated_signal_attack_1_masquerade.log": "road_attack_correlated_signal_attack_1_masquerade_030804_082640",
        "correlated_signal_attack_2_masquerade.log": "road_attack_correlated_signal_attack_2_masquerade_031128_011320",
        "correlated_signal_attack_3_masquerade.log": "road_attack_correlated_signal_attack_3_masquerade_040322_190000"
    }

    # Loop through each log file and run the processing script
    for log_file, pkl_folder in log_to_pkl.items():
        log_filepath = os.path.join(log_directory, log_file)

        # Printing the name of the dataset being processed
        print(f"Processing {log_file}...")
        
        subprocess.run(["python", "ambient_dyno_drive_basic_long.py", "--window-size", "4", "--offset", "4", "--pkl-folder", pkl_folder, log_filepath])

if __name__ == "__main__":
    main()