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
        
        all_graphs.append(G)
    

    return all_graphs

def generate_node_embeddings(all_graphs):
    all_node_embeddings = []
    for G in tqdm(all_graphs, desc="Generating Node Embeddings"):
        node2vec = Node2Vec(G, dimensions=64, walk_length=5, num_walks=10, workers=20, p=1.5, q=0.5)
        model = node2vec.fit(window=10, min_count=1, batch_words=14)
        
        node_embeddings = {}
        for node in G.nodes():
            if node in model.wv:
                node_emb = model.wv[node]
                node_embeddings[node] = node_emb
            else:
                node_embeddings[node] = np.zeros(model.vector_size)
        
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


def main():
    parser = argparse.ArgumentParser(description='Process CAN log data and generate node embeddings.')
    parser.add_argument('--window-size', type=float, required=True, help='Size of the window for slicing data.')
    parser.add_argument('--offset', type=float, required=True, help='Offset for slicing data.')
    parser.add_argument('log_filepath', type=str, help='Path to the CAN log file.')
    args = parser.parse_args()

    df = make_can_df(args.log_filepath)
    process_dataframe(df, args.window_size, args.offset)  # If you want to print slices of the dataframe
    
    # Generate graphs from data
    all_graphs = generate_graphs_from_data(df, args.window_size, args.offset)
    
    # Generate node embeddings
    all_node_embeddings = generate_node_embeddings(all_graphs)

    # Save the node embeddings to disk
    with open("node_embeddings.pkl", "wb") as f:
        pickle.dump(all_node_embeddings, f)

if __name__ == "__main__":
    main()

