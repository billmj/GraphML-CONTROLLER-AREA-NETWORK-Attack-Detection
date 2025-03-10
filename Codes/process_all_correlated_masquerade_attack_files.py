# process_all_correlated_masquerade_attack_files.py
import subprocess
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='Process CAN log data for TTW or full pipeline.')
    parser.add_argument('--mode', type=str, required=True, choices=['ttw-only', 'full-pipeline'],
                        help='Mode to run: "ttw-only" for TTW calculation, "full-pipeline" for full pipeline.')
    parser.add_argument('--window-size', type=float, required=True, help='Size of the window for slicing data.')
    parser.add_argument('--offset', type=float, required=True, help='Offset for slicing data.')
    args = parser.parse_args()

    # The directory where the new log files are located
    log_directory = "C:\\Users\\willi\\OneDrive\\Desktop\\Research\\oak_ridge_in_vehicle\\road\\attacks\\"

    # Mapping of log files to their corresponding .pkl folder names
    log_to_pkl = {
        "correlated_signal_attack_1_masquerade.log": "road_attack_correlated_signal_attack_1_masquerade_030804_082640",
        "correlated_signal_attack_2_masquerade.log": "road_attack_correlated_signal_attack_2_masquerade_031128_011320",
        "correlated_signal_attack_3_masquerade.log": "road_attack_correlated_signal_attack_3_masquerade_040322_190000"
    }

    # Determine which script to run based on the mode
    if args.mode == 'ttw-only':
        script_name = "ttw_only.py"
    elif args.mode == 'full-pipeline':
        script_name = "ambient_dyno_drive_basic_long.py"

    # Loop through each log file and run the appropriate script
    for log_file, pkl_folder in log_to_pkl.items():
        log_filepath = os.path.join(log_directory, log_file)

        # Printing the name of the dataset being processed
        print(f"Processing {log_file} in {args.mode} mode...")
        
        # Run the appropriate script
        subprocess.run([
            "python", script_name,
            "--window-size", str(args.window_size),
            "--offset", str(args.offset),
            "--pkl-folder", pkl_folder,
            log_filepath
        ])

if __name__ == "__main__":
    main()
