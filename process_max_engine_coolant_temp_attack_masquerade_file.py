import subprocess
import os
import argparse
import shutil

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Process the CAN log file for max engine coolant temp attack masquerade.')
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
    folder_name = f"max_engine_coolant_temp_w{int(args.window_size)}_o{int(args.offset)}"
    output_dir = os.path.join(output_dir, folder_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # The directory where the log file is located
    log_directory = "C:\\Users\\willi\\OneDrive\\Desktop\\Research\\oak_ridge_in_vehicle\\road\\attacks\\"

    # Specific log file and its corresponding .pkl folder name
    log_file = "max_engine_coolant_temp_attack_masquerade.log"
    pkl_folder = "road_attack_max_engine_coolant_temp_attack_masquerade_041109_063320"

    # Construct the full path to the log file
    log_filepath = os.path.join(log_directory, log_file)

    # Printing the name of the dataset being processed
    print(f"Processing {log_file}...")

    # Updated subprocess.run call to include --output-dir
    subprocess.run([
        "python", 
        "ambient_dyno_drive_basic_long.py", 
        "--window-size", str(args.window_size), 
        "--offset", str(args.offset), 
        "--pkl-folder", pkl_folder, 
        "--output-dir", output_dir,  # Pass the output directory as an argument
        log_filepath
    ])

    # After the process is complete, move or copy the CSV files to the desired output directory
    # Assuming the CSV files are generated in the current directory
    for filename in os.listdir("."):
        if filename.endswith(".csv"):
            # Move or copy the CSV file to the output directory
            shutil.move(filename, output_dir)  # Use shutil.copy if you want to copy instead of moving

if __name__ == "__main__":
    main()
