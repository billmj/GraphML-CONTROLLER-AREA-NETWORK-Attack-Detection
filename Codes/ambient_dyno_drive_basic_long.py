import os
import argparse
import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
from node2vec import Node2Vec
import pickle
from generalFunctions import unpickle
from CAN_objects.capture import MappedCapture
from math import ceil

# Function Definitions


def make_can_df(log_filepath):
    can_df = pd.read_fwf(
        log_filepath, delimiter = ' '+ '#' + '('+')',
        skiprows = 1, skipfooter=1,
        usecols = [0, 2, 3],
        dtype = {0: 'float64', 1: str, 2: str},
        names = ['time', 'pid', 'data'])

    can_df.pid = can_df.pid.apply(lambda x: int(x, 16))
    can_df.data = can_df.data.apply(lambda x: x.zfill(16))
    can_df.time = can_df.time - can_df.time.min()
    return can_df[can_df.pid <= 0x700]

        
def generate_graphs_from_data(df, window_size, offset):
    all_graphs = []
    num_slices = int((df['time'].max() - df['time'].min()) / offset)
    for i in range(num_slices):
        start_time = df['time'].min() + i * offset
        end_time = start_time + window_size
        time_slice_df = df[(df['time'] >= start_time) & (df['time'] < end_time)]
        
        G = nx.DiGraph()
        node_ids = time_slice_df['pid'].unique().tolist()
        G.add_nodes_from(node_ids)
        for j in range(len(time_slice_df) - 1):
            source = time_slice_df.iloc[j]['pid']
            target = time_slice_df.iloc[j + 1]['pid']
            if G.has_edge(source, target):
                G[source][target]['weight'] += 1
            else:
                G.add_edge(source, target, weight=1)

        if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
            print(f"Graph for slice {i+1} is empty or has no edges!")

        all_graphs.append(G)

    return all_graphs


def generate_node_embeddings(all_graphs):
    all_node_embeddings = []
    for idx, G in enumerate(tqdm(all_graphs, desc="Generating Node Embeddings")):
        node2vec = Node2Vec(G, dimensions=64, walk_length=15, num_walks=100, workers=19, p=1.5, q=0.5)
        model = node2vec.fit(window=10, min_count=1, batch_words=7)
        
        node_embeddings = {}
        for node in G.nodes():
            if node in model.wv:
                node_emb = model.wv[node]
                node_embeddings[node] = node_emb
            else:
                node_embeddings[node] = np.zeros(model.vector_size)
        
        print(f"Processed graph {idx+1}: Nodes - {G.number_of_nodes()}, Edges - {G.number_of_edges()}")

        all_node_embeddings.append(node_embeddings)
    
    return all_node_embeddings


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



def process_dataframe(df, window_size, offset):
    df_sorted = df.sort_values('time')
    df_sorted['time'] = df_sorted['time'].round(2)

    num_slices = ceil((df_sorted['time'].max() - df_sorted['time'].min()) / offset)
    for i in range(num_slices):
        start_time = df_sorted['time'].min() + i * offset
        end_time = start_time + window_size
        time_slice_label = f"Time Slice {i + 1}: ({start_time}, {end_time}]"
        time_slice_df = df_sorted[(df_sorted['time'] >= start_time) & (df_sorted['time'] < end_time)]
        print(f"{time_slice_label}\n{time_slice_df}\n")

def combine_embeddings_with_attributes(all_graphs, all_node_embeddings, time_slice_duration, mapped_capture):
    all_combined_node_embeddings = []

    # Calculate the maximum number of attributes across all nodes and all graphs
    max_attributes = max(
        [len(extract_mean_std_for_id(node, idx * time_slice_duration, (idx + 1) * time_slice_duration, mapped_capture))
         for idx, graph in enumerate(all_graphs)
         for node in graph.nodes()]
    )

    # Iterate over all graphs and their index
    for idx, G in tqdm(enumerate(all_graphs), desc="Combining Node Embeddings with Attributes"):
        current_embeddings = all_node_embeddings[idx]
        
        # Calculate start_time and end_time based on the current graph index (idx)
        start_time = idx * time_slice_duration
        end_time = (idx + 1) * time_slice_duration
        
        combined_embeddings = {}
        for node in G.nodes():
            # Get node2vec embedding for the node
            node_emb = current_embeddings[node]
            
            # Extract mean and std attributes using the mean and std function
            mean_std_dict = extract_mean_std_for_id(node, start_time, end_time, mapped_capture)
            
            # Ensure the node attribute vector is of consistent length
            attributes = list(mean_std_dict.values())
            attributes += [0] * (max_attributes - len(attributes))
            
            # Combine node embeddings and attributes
            combined_embedding = np.concatenate([node_emb, np.array(attributes)])
            
            combined_embeddings[node] = combined_embedding

        all_combined_node_embeddings.append(combined_embeddings)

    return all_combined_node_embeddings


def display_combined_embeddings(embeddings, num_to_display=5):
    """
    Display the combined embeddings for the specified number of nodes.

    Parameters:
    - embeddings: The combined node embeddings.
    - num_to_display: The number of nodes for which to display the embeddings.
    """
    for node, embedding in list(embeddings.items())[:num_to_display]:
        print(f"Node: {node}\nEmbedding: {embedding}\n")
        
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

def compute_average_embeddings(all_combined_node_embeddings, highest_signal_count):
    """
    Compute the average embeddings for each graph based on the combined node embeddings.

    Parameters:
    - all_combined_node_embeddings: List of dictionaries containing combined node embeddings for each graph.
    - highest_signal_count: Integer indicating the highest number of signals across all nodes.

    Returns:
    - concatenated_embeddings: Dictionary containing average embeddings for each graph.
    """
    # Calculate the new dimension based on 64 + (2 attributes * highest number of signals)
    new_dimension = 64 + (2 * highest_signal_count)

    # Initialize a dictionary to store concatenated embeddings
    concatenated_embeddings = {}

    # Iterate through all graphs in the capture
    for graph_embeddings in all_combined_node_embeddings:
        # Initialize variables to accumulate embeddings and count nodes
        total_embeddings = np.zeros((new_dimension,))
        num_nodes = 0

        # Iterate through node embeddings in the current graph
        for node_id, node_embedding in graph_embeddings.items():
            # Pad node embeddings to the new dimension
            padded_embeddings = np.pad(node_embedding, (0, new_dimension - len(node_embedding)), mode='constant')
            
            # Accumulate the padded embeddings
            total_embeddings += padded_embeddings
            num_nodes += 1

        # Calculate average embeddings for the current graph
        if num_nodes > 0:
            average_embeddings = total_embeddings / num_nodes
        else:
            # Handle the case where there are no nodes in the graph
            average_embeddings = np.zeros((new_dimension,))
        
        # Store the average embeddings in the dictionary
        concatenated_embeddings[len(concatenated_embeddings)] = average_embeddings

    return concatenated_embeddings

        

def get_first_graph_embedding(concatenated_embeddings):
    """
    Access the embeddings for the first graph.
    
    Parameters:
    - concatenated_embeddings: Dictionary containing the embeddings for each graph.
    
    Returns:
    - first_graph_embedding: Embeddings for the first graph.
    """
    return concatenated_embeddings[0]

def create_dataframe_from_embeddings(concatenated_embeddings):
    """
    Create a DataFrame from the concatenated embeddings.
    
    Parameters:
    - concatenated_embeddings: Dictionary containing the embeddings for each graph.
    
    Returns:
    - df_benign: DataFrame with embeddings and labels.
    """
    # Create a list to store the embeddings and labels
    data = []

    # Iterate through the benign embeddings
    for graph_id, embedding in concatenated_embeddings.items():
        # Append the embedding and label to the data list
        data.append((embedding, '0'))

    # Create a DataFrame
    df_benign = pd.DataFrame(data, columns=['Embedding', 'Label'])

    return df_benign



def save_dataframe_to_csv(df, window_size, offset, prefix="benign"):
    """
    Save the given dataframe to a CSV file. The filename will be constructed using
    the given prefix and the window_size and offset parameters.

    Parameters:
    - df: DataFrame to save.
    - window_size: Window size used in the experiment.
    - offset: Offset used in the experiment.
    - prefix: Prefix for the filename.
    """
    filename = f"{prefix}_window_size_{window_size}_offset_{offset}.csv"
    df.to_csv(filename, index=False)

def main():
    parser = argparse.ArgumentParser(description='Process CAN log data and generate node embeddings.')
    parser.add_argument('--window-size', type=float, required=True, help='Size of the window for slicing data.')
    parser.add_argument('--offset', type=float, required=True, help='Offset for slicing data.')
    parser.add_argument('log_filepath', type=str, help='Path to the CAN log file.')
    args = parser.parse_args()

    # Read the CAN log and process it
    df = make_can_df(args.log_filepath)
    process_dataframe(df, args.window_size, args.offset)

    # Generate graphs from data
    all_graphs = generate_graphs_from_data(df, args.window_size, args.offset)

    # Generate node embeddings
    all_node_embeddings = generate_node_embeddings(all_graphs)

    # Load mapped_capture for attribute extraction
    mapped_capture = load_mapped_capture()

    # Combine the embeddings with attributes
    all_combined_node_embeddings = combine_embeddings_with_attributes(all_graphs, all_node_embeddings, args.window_size, mapped_capture)

    # Display the first few combined embeddings for the first graph
    print("Displaying first few combined embeddings for the first graph:")
    display_combined_embeddings(all_combined_node_embeddings[0])

    # Find the node with the highest number of signals
    node, highest_signal_count = find_node_with_highest_signals(mapped_capture)
    print(f"Node with Highest Signals (Node ID, Signal Count): ({node}, {highest_signal_count})")
    
    # Compute average embeddings
    concatenated_embeddings = compute_average_embeddings(all_combined_node_embeddings, highest_signal_count)

    # Print the number of graphs processed and the keys in concatenated_embeddings
    print("Number of graphs processed:", len(concatenated_embeddings))
    print("Keys in concatenated_embeddings:", list(concatenated_embeddings.keys()))

    

    # Use the function to create the DataFrame
    df_benign = create_dataframe_from_embeddings(concatenated_embeddings)
    
    # Display the first few rows of the DataFrame
    print(df_benign.head())

    # Display the last few rows of the DataFrame
    print(df_benign.iloc[-5:])
    
    # Save the DataFrame to a CSV file
    save_dataframe_to_csv(df_benign, args.window_size, args.offset)
    
    # Save the combined embeddings to disk
    with open("combined_node_embeddings.pkl", "wb") as f:
        pickle.dump(all_combined_node_embeddings, f)

if __name__ == "__main__":
    main()




