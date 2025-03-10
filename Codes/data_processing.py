# data_processing.py
from imports import *

def make_can_df(log_filepath):
    """
    Read and preprocess a CAN log file into a DataFrame.

    Parameters:
    - log_filepath: Path to the CAN log file.

    Returns:
    - can_df: Preprocessed DataFrame containing CAN data.
    """
    # Read the CAN log file into a DataFrame
    can_df = pd.read_fwf(
        log_filepath, delimiter=' ' + '#' + '(' + ')',
        skiprows=1, skipfooter=1,
        usecols=[0, 2, 3],
        dtype={0: 'float64', 1: str, 2: str},
        names=['time', 'pid', 'data']
    )

    # Convert the 'pid' column from hexadecimal to integer
    can_df.pid = can_df.pid.apply(lambda x: int(x, 16))

    # Pad the 'data' column to ensure it has 16 characters
    can_df.data = can_df.data.apply(lambda x: x.zfill(16))

    # Normalize the 'time' column to start from 0
    can_df.time = can_df.time - can_df.time.min()

    # Filter out rows where 'pid' is greater than 0x700
    return can_df[can_df.pid <= 0x700]

def generate_graphs_from_data(df, window_size, offset):
    """
    Generate graphs from CAN data using sliding windows based on time.

    Parameters:
    - df: DataFrame containing CAN log data.
    - window_size: Size of the sliding window (in seconds).
    - offset: Offset between consecutive windows (in seconds).

    Returns:
    - all_graphs: List of graphs generated for each window.
    """
    all_graphs = []  # List to store graphs for each window
    num_slices = int((df['time'].max() - df['time'].min()) // offset + 1)  # Number of windows
    
    for i in range(num_slices):
        # Define the time window
        start_time = df['time'].min() + i * offset
        end_time = start_time + window_size
        time_slice_df = df[(df['time'] >= start_time) & (df['time'] < end_time)]
        
        # Create a directed graph for the current window
        G = nx.DiGraph()
        node_ids = time_slice_df['pid'].unique().tolist()
        G.add_nodes_from(node_ids)
        
        # Add edges based on message sequences
        for j in range(len(time_slice_df) - 1):
            source = time_slice_df.iloc[j]['pid']
            target = time_slice_df.iloc[j + 1]['pid']
            if G.has_edge(source, target):
                G[source][target]['weight'] += 1  # Increment edge weight if it exists
            else:
                G.add_edge(source, target, weight=1)  # Add new edge with weight 1

        # Handle empty graphs
        if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
            print(f"Graph for slice {i+1} is empty or has no edges!")
            G.add_node(0)  # Add a node with value 0 to represent an empty graph

        all_graphs.append(G)  # Store the graph
        print(f"Processed slice {i+1}/{num_slices}, Graph Nodes: {G.number_of_nodes()}, Graph Edges: {G.number_of_edges()}")

    return all_graphs

def process_dataframe(df, window_size, offset):
    """
    Process the DataFrame into sliding windows and display the slices.

    Parameters:
    - df: DataFrame containing CAN log data.
    - window_size: Size of the sliding window (in seconds).
    - offset: Offset between consecutive windows (in seconds).
    """
    df_sorted = df.sort_values('time')
    df_sorted['time'] = df_sorted['time'].round(2)

    num_slices = int((df_sorted['time'].max() - df_sorted['time'].min()) // offset + 1)
    for i in range(num_slices):
        start_time = df_sorted['time'].min() + i * offset
        end_time = start_time + window_size
        time_slice_label = f"Time Slice {i + 1}: ({start_time}, {end_time}]"
        time_slice_df = df_sorted[(df_sorted['time'] >= start_time) & (df_sorted['time'] < end_time)]
        print(f"{time_slice_label}\n{time_slice_df}\n")

def generate_graphs_from_data_samples(df, window_size, offset):
    """
    Generate graphs from CAN data using sliding windows based on a fixed number of samples.

    Parameters:
    - df: DataFrame containing CAN log data.
    - window_size: Number of samples in each window.
    - offset: Number of samples between consecutive windows.

    Returns:
    - all_graphs: List of graphs generated for each window.
    """
    all_graphs = []  # List to store graphs for each window
    num_samples = len(df)  # Total number of samples in the DataFrame
    
    for start in range(0, num_samples, offset):
        end = start + window_size
        if end > num_samples:
            break  # Stop if the window exceeds the number of samples
        
        # Extract the slice of the DataFrame
        sample_slice_df = df.iloc[start:end]
        
        # Create a directed graph for the current window
        G = nx.DiGraph()
        node_ids = sample_slice_df['pid'].unique().tolist()
        G.add_nodes_from(node_ids)
        
        # Add edges based on message sequences
        for j in range(len(sample_slice_df) - 1):
            source = sample_slice_df.iloc[j]['pid']
            target = sample_slice_df.iloc[j + 1]['pid']
            if G.has_edge(source, target):
                G[source][target]['weight'] += 1  # Increment edge weight if it exists
            else:
                G.add_edge(source, target, weight=1)  # Add new edge with weight 1

        # Handle empty graphs
        if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
            print(f"Graph for slice starting at sample {start} is empty or has no edges!")
            G.add_node(0)  # Add a node with value 0 to represent an empty graph

        all_graphs.append(G)  # Store the graph
        print(f"Processed slice starting at sample {start}, Graph Nodes: {G.number_of_nodes()}, Graph Edges: {G.number_of_edges()}")

    return all_graphs

def process_dataframe_samples(df, window_size, offset):
    """
    Process the DataFrame into sliding windows based on a fixed number of samples and display the slices.

    Parameters:
    - df: DataFrame containing CAN log data.
    - window_size: Number of samples in each window.
    - offset: Number of samples between consecutive windows.
    """
    df_sorted = df.sort_values('time')
    num_samples = len(df_sorted)  # Total number of samples in the DataFrame
    
    for start in range(0, num_samples, offset):
        end = start + window_size
        if end > num_samples:
            break  # Stop if the window exceeds the number of samples
        
        # Extract the slice of the DataFrame
        sample_slice_df = df_sorted.iloc[start:end]
        print(f"Sample Slice: ({start}, {end})\n{sample_slice_df}\n")
