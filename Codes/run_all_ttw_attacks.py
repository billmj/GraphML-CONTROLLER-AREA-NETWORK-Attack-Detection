#run_all_ttw_attacks.py

import subprocess

# Define the attack types and their corresponding script names
attack_types = {
    "correlated_signal": "process_all_correlated_masquerade_attack_files.py",
    "max_speedometer": "process_all_max_speedometer_attack_files.py",
    "reverse_light_off": "process_all_reverse_light_off_attack_files.py",
    "reverse_light_on": "process_all_reverse_light_on_attack_files.py",
    "max_engine_coolant_temp": "process_max_engine_coolant_temp_attack_masquerade_file.py"
}

# Define the range of window sizes and offsets
window_sizes = range(2, 16)  # From 2 to 15

# Loop through each attack type
for attack_name, script_name in attack_types.items():
    print(f"Running TTW calculations for {attack_name} attack...")
    
    # Loop through each window size and offset
    for window_size in window_sizes:
        for offset in range(1, window_size + 1):
            # Build the command
            command = [
                "python", script_name,
                "--mode", "ttw-only",
                "--window-size", str(window_size),
                "--offset", str(offset)
            ]
            
            # Print the command for debugging
            print("Running:", " ".join(command))
            
            # Run the command
            subprocess.run(command)
    
    print(f"Finished TTW calculations for {attack_name} attack.\n")

print("All TTW calculations completed!")