import pandas as pd
import numpy as np
from CAN_objects.capture import MappedCapture

def extract_mean_std(df):
    """
    Extract the mean and standard deviation for each column in a DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing signal data.

    Returns:
    list: Two lists containing mean and standard deviation values for each column.
    """
    return [df.mean().values, df.std().values]

def extract_mean_std_for_id(node_id, start_time, end_time, mapped_capture):
    """
    Extract the mean and standard deviation for a specific ID within a time window.

    Parameters:
    node_id (int): The ID of the node to analyze.
    start_time (int): The start time of the time window.
    end_time (int): The end time of the time window.
    mapped_capture (MappedCapture): The mapped capture data.

    Returns:
    dict: A dictionary containing mean and standard deviation values for each signal in the time window.
    """
    mp = mapped_capture.mapped_payload_dict[node_id]
    mask = (mp.times >= start_time) & (mp.times < end_time)
    sub_df = pd.DataFrame()
    for signal in mp.signal_list:
        signal_values = signal.values[mask]
        interpolated_values = pd.Series(signal_values).interpolate(method='linear', limit_area='inside').values
        sub_df[signal.name] = interpolated_values
    mean_std_dict = {}
    for col, (mean, std) in zip(sub_df.columns, zip(sub_df.mean().values, sub_df.std().values)):
        mean_std_dict[f"{col}_mean"] = round(mean, 2)
        mean_std_dict[f"{col}_std"] = round(std, 2)
    return mean_std_dict

def extract_signal_data_for_id(node_id, mapped_capture, window_length=10):
    """
    Extract signal data for a specific ID within customizable time windows.

    Parameters:
    node_id (int): The ID of the node to analyze.
    mapped_capture (MappedCapture): The mapped capture data.
    window_length (int): The length of the time window in seconds. Default is 10 seconds.

    Returns:
    pd.DataFrame: A DataFrame containing signal data for the ID.
    """
    mp = mapped_capture.mapped_payload_dict[node_id]
    data_dict = {}
    time_windows = range(0, int(mp.times[-1]), window_length)
    for window_start in time_windows:
        window_end = window_start + window_length
        sub_df = pd.DataFrame(index=range(10))  # Assuming 10 samples in each time window
        for signal in mp.signal_list:
            mask = (signal.times >= window_start) & (signal.times < window_end)
            signal_values = signal.values[mask]
            interpolated_values = pd.Series(signal_values).interpolate(method='linear', limit_area='inside').values
            interpolated_values = [interpolated_values[i] if i < len(interpolated_values) else None for i in range(10)]
            interpolated_values = pd.Series(interpolated_values).interpolate(method='linear', limit_area='inside').values
            sub_df[signal.name] = interpolated_values
        data_dict[(window_start, window_end)] = sub_df
    df = pd.concat(data_dict, axis=1)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df

def extract_signal_data_for_all_ids(ids_to_explore, mapped_capture, window_length=10):
    """
    Extract signal data for a list of IDs within customizable time windows.

    Parameters:
    ids_to_explore (list): A list of node IDs to analyze.
    mapped_capture (MappedCapture): The mapped capture data.
    window_length (int): The length of the time window in seconds. Default is 10 seconds.

    Returns:
    dict: A dictionary containing signal data DataFrames for each ID.
    """
    signal_data_dict = {}
    for id_to_explore in ids_to_explore:
        signal_data = extract_signal_data_for_id(id_to_explore, mapped_capture, window_length)
        signal_data_dict[id_to_explore] = signal_data
    return signal_data_dict
