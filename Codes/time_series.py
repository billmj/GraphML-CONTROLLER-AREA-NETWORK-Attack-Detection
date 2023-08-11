import pandas as pd
import numpy as np

# List of IDs to explore
ids_to_explore = [6, 14, 37, ...]  # List of IDs you want to explore

# Function to extract mean and std for each column in a DataFrame
def extract_mean_std(df):
    return [df.mean().values, df.std().values]

# Loop through each ID and extract the signal values
for id_to_explore in ids_to_explore:
    mp = mapped_capture.mapped_payload_dict[id_to_explore]

    # Display the information about the ID and its signals
    print(f"ID {id_to_explore}")

    # Define the time windows (every 10 seconds)
    time_windows = range(0, int(mp.times[-1]), 10)

    # Loop through the signal_list and extract information for each time window
    for window_start in time_windows:
        window_end = window_start + 10
        print(f"{window_start} - {window_end}")

        # Create a sub-DataFrame for the current time window
        sub_df = pd.DataFrame(index=range(10))  # Assuming 10 samples in each time window

        for signal in mp.signal_list:
            # Get the indices for the current time window
            mask = (signal.times >= window_start) & (signal.times < window_end)

            # Get the values within the current time window and interpolate for consistent time resolution
            signal_values = signal.values[mask]
            interpolated_values = pd.Series(signal_values).interpolate(method='linear', limit_area='inside').values

            # Ensure that interpolated values match the length of the DataFrame
            interpolated_values = [interpolated_values[i] if i < len(interpolated_values) else None for i in range(10)]
            interpolated_values = pd.Series(interpolated_values).interpolate(method='linear', limit_area='inside').values

            # Add the interpolated signal values to the sub-DataFrame
            sub_df[signal.name] = interpolated_values

        # Compute the mean and std for the sub-DataFrame
        mean_std_array = extract_mean_std(sub_df)

        # Display the mean and std for each signal in the current time window
        for signal_name, (mean_val, std_val) in zip(sub_df.columns, zip(*mean_std_array)):
            print(f"{signal_name} [{mean_val:.2f}, {std_val:.2f}]")

        print()
