import subprocess
import os

def main():
    # The directory where your log files are located
    log_directory = "C:\\Users\\willi\\OneDrive\\Desktop\\Research\\oak_ridge_in_vehicle\\road\\ambient\\"

    # List of log files you want to process
    log_files = [
        "ambient_dyno_drive_basic_short.log",
        "ambient_dyno_drive_benign_anomaly.log",
        "ambient_dyno_drive_extended_long.log",
        "ambient_dyno_drive_extended_short.log",
        "ambient_dyno_drive_radio_infotainment.log",
        "ambient_dyno_drive_winter.log",
        "ambient_dyno_exercise_all_bits.log",
        "ambient_dyno_idle_radio_infotainment.log",
        "ambient_dyno_reverse.log"
    ]

    # Loop through each log file and run the processing script
    for log_file in log_files:
        log_filepath = os.path.join(log_directory, log_file)
        
        # Printing the name of the dataset being processed
        print(f"Processing {log_file}...")
        
        subprocess.run(["python", "ambient_dyno_drive_basic_long.py", "--window-size", "10", "--offset", "10", log_filepath])

if __name__ == "__main__":
    main()
