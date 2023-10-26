import subprocess
import os

def main():
    # The directory where our log files are located
    log_directory = "C:\\Users\\willi\\OneDrive\\Desktop\\Research\\oak_ridge_in_vehicle\\road\\ambient\\"

    # Dictionary mapping log files to their respective pkl folders
    log_to_pkl = {
        "ambient_dyno_drive_basic_short.log": "road_ambient_dyno_drive_basic_short_020822_030640",
        "ambient_dyno_drive_benign_anomaly.log": "road_ambient_dyno_drive_benign_anomaly_030804_082640",
        "ambient_dyno_drive_extended_long.log": "road_ambient_dyno_drive_extended_long_040716_134640",
        "ambient_dyno_drive_extended_short.log": "road_ambient_dyno_drive_extended_short_021215_195320",
        "ambient_dyno_drive_radio_infotainment.log": "road_ambient_dyno_drive_radio_infotainment_041109_063320",
        "ambient_dyno_drive_winter.log": "road_ambient_dyno_drive_winter_030410_144000",
        "ambient_dyno_exercise_all_bits.log": "road_ambient_dyno_exercise_all_bits_030410_144000",
        "ambient_dyno_idle_radio_infotainment.log": "road_ambient_dyno_idle_radio_infotainment_030410_144000",
        "ambient_dyno_reverse.log": "road_ambient_dyno_reverse_040322_190000"
    }

    # Loop through each log file and run the processing script
    for log_file, pkl_folder in log_to_pkl.items():
        log_filepath = os.path.join(log_directory, log_file)
        
        # Printing the name of the dataset being processed
        print(f"Processing {log_file}...")
        
        subprocess.run(["python", "ambient_dyno_drive_basic_long.py", "--window-size", "10", "--offset", "10", "--pkl-folder", pkl_folder, log_filepath])

if __name__ == "__main__":
    main()
