# utils.py
from imports import *


from generalFunctions import unpickle
from CAN_objects.capture import MappedCapture

def load_mapped_capture():
    # Define the file path for the CAN capture
    cancap_filepath = os.path.join("C:\\Users\\willi\\Dropbox\\data-cancaptures", 
                                  "road_ambient_dyno_drive_basic_long_050305_002000", "capture.pkl")
    cancap = unpickle(cancap_filepath)
    
    # Define the file path for the ground truth DBC file
    ground_truth_dbc_fpath = os.path.join("C:\\Users\\willi\\Dropbox\\data-cancaptures\\DBC", "anonymized_020822_030640.dbc")
    mapped_capture = MappedCapture.init_from_dbc(cancap, ground_truth_dbc_fpath)
    
    return mapped_capture

def extract_mean_std_for_id(node_id, start_time, end_time, mapped_capture):
    """
    Function to extract the mean and standard deviation for a specific ID in a given time window.

    Parameters:
    - node_id: The ID of the node to extract data for.
    - start_time: The start time of the time window.
    - end_time: The end time of the time window.
    - mapped_capture: The MappedCapture object containing the data.

    Returns:
    - mean_std_dict: A dictionary containing the mean and standard deviation for each signal of the node within the specified time window.
    """
    # Extract the mapped payload for the given ID
    mp = mapped_capture.mapped_payload_dict[node_id]

    # Get the mask for the current time window
    mask = (mp.times >= start_time) & (mp.times < end_time)

    # Create a sub-DataFrame for the current time window
    sub_df = pd.DataFrame()

    for signal in mp.signal_list:
        signal_values = signal.values[mask]
        interpolated_values = pd.Series(signal_values).interpolate(method='linear', limit_area='inside').values
        sub_df[signal.name] = interpolated_values

    # Compute the mean and std for the sub-DataFrame
    mean_std_dict = {}
    for col, (mean, std) in zip(sub_df.columns, zip(sub_df.mean().values, sub_df.std().values)):
        mean_std_dict[f"{col}_mean"] = round(mean, 2)
        mean_std_dict[f"{col}_std"] = round(std, 2)

    return mean_std_dict


def find_node_with_highest_signals(mapped_capture):
    """
    Find and return the node with the highest number of signals from the mapped_capture object.

    Parameters:
    - mapped_capture: The MappedCapture object containing the data.

    Returns:
    - node_with_highest_signals: The node ID with the highest number of signals.
    - highest_signal_count: The highest signal count.
    """
    # Initialize variables to keep track of the node with the highest number of signals
    highest_signal_count = 0
    node_with_highest_signals = None

    # Iterate through the node IDs
    for node_id in mapped_capture.mapped_payload_dict.keys():
        # Get the mapped payload for the current node
        mp = mapped_capture.mapped_payload_dict[node_id]
        
        # Count the number of signals for this node
        signal_count = len(mp.signal_list)
        
        # Check if this node has more signals than the current highest count
        if signal_count > highest_signal_count:
            highest_signal_count = signal_count
            node_with_highest_signals = node_id

    return node_with_highest_signals, highest_signal_count

def get_first_graph_embedding(concatenated_embeddings):
    """
    Access the embeddings for the first graph.
    
    Parameters:
    - concatenated_embeddings: Dictionary containing the embeddings for each graph.
    
    Returns:
    - first_graph_embedding: Embeddings for the first graph.
    """
    return concatenated_embeddings[0]

def save_dataframe_to_csv(df, window_size, offset, log_filename, prefix="benign"):
    """
    Save the given dataframe to a CSV file. The filename will be constructed using
    the given prefix, the name of the log file, and the window_size and offset parameters.

    Parameters:
    - df: DataFrame to save.
    - window_size: Window size used in the experiment.
    - offset: Offset used in the experiment.
    - log_filename: Name of the log file being processed.
    - prefix: Prefix for the filename.
    """
    # Extract the base name of the log file without the extension
    base_log_filename = os.path.basename(log_filename).split('.')[0]
    
    filename = f"{prefix}_{base_log_filename}_window_size_{window_size}_offset_{offset}.csv"
    df.to_csv(filename, index=False)

