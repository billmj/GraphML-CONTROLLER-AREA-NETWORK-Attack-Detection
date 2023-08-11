# data_preprocessing.py

import pandas as pd

# Function to read and preprocess the raw CAN log file
def make_can_df(log_filepath):
    # ...

# Read the log file and parse the contents into a DataFrame
df = make_can_df(file)

# Sort the DataFrame by the 'time' column and round the 'time' values to 2 decimal places
df_sorted = df.sort_values('time')
df_sorted['time'] = df_sorted['time'].round(2)

# Divide the sorted DataFrame into time slices of duration 10.0 and print each slice along with its label
time_slice_duration = 10.0
num_slices = int(df_sorted['time'].max() / time_slice_duration) + 1
for i in range(num_slices):
    start_time = i * time_slice_duration
    end_time = (i + 1) * time_slice_duration
    time_slice_label = f"Time Slice {i + 1}: ({start_time}, {end_time}]"
    time_slice_df = df_sorted[(df_sorted['time'] > start_time) & (df_sorted['time'] <= end_time)]
    print(f"{time_slice_label}\n{time_slice_df}\n")
