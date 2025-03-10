#process_all_max_speedometer_attack_files.py

import subprocess
import os
import argparse

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description='Process multiple CAN log files for max speedometer attack masquerade.')
    parser.add_argument('--mode', type=str, required=True, choices=['ttw-only', 'full-pipeline'],
                        help='Mode to run: "ttw-only" for TTW calculation, "full-pipeline" for full pipeline.')
    parser.add_argument('--window-size', type=float, required=True, help='Size of the window for slicing data.')
    parser.add_argument('--offset', type=float, required=True, help='Offset for slicing data.')
    parser.add_argument('--output-dir', type=str, default="C:\\Users\\willi\\CAN_experiments\\Specific_Attack_Types",
                        help='Directory to save processed files. If not provided, the default output directory will be used.')

    args = parser.parse_args()

    # Create a folder based on window size and offset inside the output directory
    folder_name = f"max_speedometer_w{int(args.window_size)}_o{int(args.offset)}"
    output_dir = os.path.join(args.output_dir, folder_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # The directory where the log files are located
    log_directory = "C:\\Users\\willi\\OneDrive\\Desktop\\Research\\oak_ridge_in_vehicle\\road\\attacks\\"

    # Mapping of log files to their corresponding .pkl folder names
    log_to_pkl = {
        "max_speedometer_attack_1_masquerade.log": "road_attack_max_speedometer_attack_1_masquerade_060215_054000",
        "max_speedometer_attack_2_masquerade.log": "road_attack_max_speedometer_attack_2_masquerade_060611_002640",
        "max_speedometer_attack_3_masquerade.log": "road_attack_max_speedometer_attack_3_masquerade_061004_181320"
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

        # Prepare the command to run
        command = [
            "python", script_name,
            "--window-size", str(args.window_size),
            "--offset", str(args.offset),
            "--pkl-folder", pkl_folder,
            log_filepath
        ]

        # Add --output-dir only in full-pipeline mode
        if args.mode == 'full-pipeline':
            command.extend(["--output-dir", output_dir])

        # Run the appropriate script
        subprocess.run(command)

if __name__ == "__main__":
    main()
