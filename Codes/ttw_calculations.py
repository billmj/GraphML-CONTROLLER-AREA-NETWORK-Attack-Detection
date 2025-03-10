#ttw_calculations.py

from imports import *

def calculate_ttw(df, window_size, offset):
    """
    Calculate Time-to-Window (TTW) for each sliding window.

    Parameters:
    - df: DataFrame containing CAN log data.
    - window_size: Size of the sliding window (in seconds).
    - offset: Offset between consecutive windows (in seconds).

    Returns:
    - all_graphs: List of graphs generated for each window.
    - ttws: List of TTW values for each window.
    - avg_ttw: Average TTW across all windows.
    """
    all_graphs = []  # List to store graphs for each window
    ttws = []  # List to store TTW for each window
    num_slices = int((df['time'].max() - df['time'].min()) // offset + 1)  # Estimate Number of Windows
    
    for i in range(num_slices):
        start_time_slice = time.time()  # Start timing for this window using time.time()
        
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
        
        end_time_slice = time.time()  # End timing for this window
        ttws.append(end_time_slice - start_time_slice)  # Store TTW for this window
        
        # Print progress and TTW for the current window
        print(f"Processed slice {i+1}/{num_slices}, Graph Nodes: {G.number_of_nodes()}, Graph Edges: {G.number_of_edges()}, TTW: {ttws[-1]:.4f} seconds")

    # Calculate average TTW
    avg_ttw = sum(ttws) / len(ttws) if ttws else 0
    print(f"Average TTW: {avg_ttw:.4f} seconds")
    
    return all_graphs, ttws, avg_ttw  # Return graphs, TTW list, and average TTW

def save_ttw_results(ttws, avg_ttw, log_filename, window_size, offset):
    """
    Save TTW results to a CSV file in a dedicated folder.

    Parameters:
    - ttws: List of TTW values for each window.
    - avg_ttw: Average TTW across all windows.
    - log_filename: Name of the log file being processed.
    - window_size: Window size used in the experiment.
    - offset: Offset used in the experiment.
    """
    # Create a folder for TTW results if it doesn't exist
    output_folder = "ttw_results"
    os.makedirs(output_folder, exist_ok=True)
    
    # Define the output file name
    output_file = os.path.join(output_folder, f"ttw_{os.path.basename(log_filename).split('.')[0]}_w{window_size}_o{offset}.csv")
    
    # Save the results
    results = {
        "Window": list(range(1, len(ttws) + 1)),
        "TTW (s)": ttws
    }
    results_df = pd.DataFrame(results)
    results_df.loc[len(results_df)] = ["Average", avg_ttw]  # Add average TTW as the last row
    results_df.to_csv(output_file, index=False)
    print(f"TTW results saved to {output_file}")