import subprocess
import os
import argparse
import shutil

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Process multiple CAN log files for reverse light off attack masquerade.')
    parser.add_argument('--window-size', type=float, required=True, help='Size of the window for slicing data.')
    parser.add_argument('--offset', type=float, required=True, help='Offset for slicing data.')
    # New: Optional argument for specifying the output directory
    parser.add_argument('--output-dir', type=str, default="C:\\Users\\willi\\CAN_experiments\\Specific_Attack_Types", help='Directory to save processed files. If not provided, the default output directory will be used.')

    args = parser.parse_args()

    # Create a folder based on window size and offset inside the output directory
    folder_name = f"reverse_light_off_w{int(args.window_size)}_o{int(args.offset)}"
    output_dir = os.path.join(args.output_dir, folder_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # The directory where the log files are located
    log_directory = "C:\\Users\\willi\\OneDrive\\Desktop\\Research\\oak_ridge_in_vehicle\\road\\attacks\\"

    # Mapping of log files to their corresponding .pkl folder names
    log_to_pkl = {
        "reverse_light_off_attack_1_masquerade.log": "road_attack_reverse_light_off_attack_1_masquerade_080110_162000",
        "reverse_light_off_attack_2_masquerade.log": "road_attack_reverse_light_off_attack_2_masquerade_080505_110640",
        "reverse_light_off_attack_3_masquerade.log": "road_attack_reverse_light_off_attack_3_masquerade_080829_045320"
    }

    # Loop through each log file and run the processing script
    for log_file, pkl_folder in log_to_pkl.items():
        log_filepath = os.path.join(log_directory, log_file)

        # Create a folder based on window size and offset inside the output directory
        window_offset_folder = f"reverse_light_off_w{int(args.window_size)}_o{int(args.offset)}"
        output_dir = os.path.join(args.output_dir, window_offset_folder)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Printing the name of the dataset being processed
        print(f"Processing {log_file}...")

        # Updated subprocess.run call to include --output-dir
        subprocess.run([
            "python", "ambient_dyno_drive_basic_long.py", 
            "--window-size", str(args.window_size), 
            "--offset", str(args.offset), 
            "--pkl-folder", pkl_folder, 
            log_filepath,
            "--output-dir", output_dir  # Pass the output directory to the processing script
        ])

if __name__ == "__main__":
    main()
