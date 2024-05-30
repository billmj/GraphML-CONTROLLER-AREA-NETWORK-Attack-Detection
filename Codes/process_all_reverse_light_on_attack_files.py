import subprocess
import os
import argparse
import shutil

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Process multiple CAN log files for reverse light on attack masquerade.')
    parser.add_argument('--window-size', type=float, required=True, help='Size of the window for slicing data.')
    parser.add_argument('--offset', type=float, required=True, help='Offset for slicing data.')
    
    # New: Optional argument for specifying the output directory
    parser.add_argument('--output-dir', type=str, default=None, help='Directory to save processed files. If not provided, the default output directory will be used.')

    args = parser.parse_args()

    # Ensure the output directory exists
    if args.output_dir is None:
        # Define the default output directory
        default_output_dir = "C:\\Users\\willi\\CAN_experiments\\Specific_Attack_Types"
        output_dir = default_output_dir
    else:
        output_dir = args.output_dir

    # Create a folder based on window size and offset inside the output directory
    folder_name = f"reverse_light_on_w{int(args.window_size)}_o{int(args.offset)}"
    output_dir = os.path.join(output_dir, folder_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # The directory where the log files are located
    log_directory = "C:\\Users\\willi\\OneDrive\\Desktop\\Research\\oak_ridge_in_vehicle\\road\\attacks\\"

    # Mapping of log files to their corresponding .pkl folder names
    log_to_pkl = {
        "reverse_light_on_attack_1_masquerade.log": "road_attack_reverse_light_on_attack_1_masquerade_091205_030000",
        "reverse_light_on_attack_2_masquerade.log": "road_attack_reverse_light_on_attack_2_masquerade_100330_214640",
        "reverse_light_on_attack_3_masquerade.log": "road_attack_reverse_light_on_attack_3_masquerade_100724_153320"
    }

    # Loop through each log file and run the processing script
    for log_file, pkl_folder in log_to_pkl.items():
        log_filepath = os.path.join(log_directory, log_file)

        # Printing the name of the dataset being processed
        print(f"Processing {log_file}...")

        # Updated subprocess.run call to include --output-dir
        subprocess.run([
            "python", "ambient_dyno_drive_basic_long.py", 
            "--window-size", str(args.window_size), 
            "--offset", str(args.offset), 
            "--pkl-folder", pkl_folder, 
            "--output-dir", output_dir,  # Pass the output directory to the processing script
            log_filepath
        ])

if __name__ == "__main__":
    main()
