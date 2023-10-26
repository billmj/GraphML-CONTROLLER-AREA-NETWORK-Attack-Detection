from imports import *

from data_processing import make_can_df, generate_graphs_from_data, process_dataframe
from embedding_generation import (generate_node_embeddings, combine_embeddings_with_attributes, 
                                 compute_average_embeddings, create_dataframe_from_embeddings, display_combined_embeddings)
from utils import load_mapped_capture, find_node_with_highest_signals, save_dataframe_to_csv

def main():
    parser = argparse.ArgumentParser(description='Process CAN log data and generate node embeddings.')
    parser.add_argument('--window-size', type=float, required=True, help='Size of the window for slicing data.')
    parser.add_argument('--offset', type=float, required=True, help='Offset for slicing data.')
    parser.add_argument('--pkl-folder', type=str, required=True, help='Folder name for the .pkl file.')
    parser.add_argument('log_filepath', type=str, help='Path to the CAN log file.')
    args = parser.parse_args()

    # Read the CAN log and process it
    df = make_can_df(args.log_filepath)
    process_dataframe(df, args.window_size, args.offset)

    # Generate graphs from data
    all_graphs = generate_graphs_from_data(df, args.window_size, args.offset)

    # Generate node embeddings
    all_node_embeddings = generate_node_embeddings(all_graphs)

    # Load mapped_capture for attribute extraction using the specified pkl folder
    mapped_capture = load_mapped_capture(args.pkl_folder)

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
    save_dataframe_to_csv(df_benign, args.window_size, args.offset, args.log_filepath)

    # Save the combined embeddings to disk
    with open("combined_node_embeddings.pkl", "wb") as f:
        pickle.dump(all_combined_node_embeddings, f)

if __name__ == "__main__":
    main()
