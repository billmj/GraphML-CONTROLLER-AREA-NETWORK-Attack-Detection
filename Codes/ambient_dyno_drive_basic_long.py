#ambient_dyno_drive_basic_long.py

from imports import *
from data_processing import make_can_df, process_dataframe
from embedding_generation import (generate_node_embeddings, combine_embeddings_with_attributes, 
                                 compute_average_embeddings, create_dataframe_from_embeddings, display_combined_embeddings)
from utils import load_mapped_capture, find_node_with_highest_signals, save_dataframe_to_csv
from ttw_calculations import calculate_ttw, save_ttw_results

def main():
    parser = argparse.ArgumentParser(description='Process CAN log data and generate node embeddings.')
    parser.add_argument('--window-size', type=float, required=True, help='Size of the window for slicing data.')
    parser.add_argument('--offset', type=float, required=True, help='Offset for slicing data.')
    parser.add_argument('--pkl-folder', type=str, required=True, help='Folder name for the .pkl file.')
    parser.add_argument('log_filepath', type=str, help='Path to the CAN log file.')
    parser.add_argument('--output-dir', type=str, default=".", help='Directory to save output files.')
    args = parser.parse_args()

    # Read the CAN log and process it
    df = make_can_df(args.log_filepath)
    process_dataframe(df, args.window_size, args.offset)

    # Generate graphs and calculate TTW
    all_graphs, ttws, avg_ttw = calculate_ttw(df, args.window_size, args.offset)

    # Save TTW results to a CSV file
    ttw_output_file = os.path.join(args.output_dir, f"ttw_results_{os.path.basename(args.log_filepath).split('.')[0]}_w{args.window_size}_o{args.offset}.csv")
    save_ttw_results(ttws, avg_ttw, args.log_filepath, args.window_size, args.offset)

    # Continue with the rest of the pipeline (e.g., node embeddings, etc.)
    all_node_embeddings = generate_node_embeddings(all_graphs)

    # Load mapped_capture for attribute extraction using the specified pkl folder
    mapped_capture = load_mapped_capture(args.pkl_folder)
    
    # Combine the embeddings with attributes
    all_combined_node_embeddings = combine_embeddings_with_attributes(all_graphs, all_node_embeddings, args.window_size, mapped_capture)
    
    # Find the node with the highest number of signals
    node, highest_signal_count = find_node_with_highest_signals(mapped_capture)
    print(f"Node with Highest Signals (Node ID, Signal Count): ({node}, {highest_signal_count})")

    # Compute average embeddings
    concatenated_embeddings = compute_average_embeddings(all_combined_node_embeddings, highest_signal_count)

    # Create and save the final DataFrame
    df_benign = create_dataframe_from_embeddings(concatenated_embeddings, args.log_filepath, args.offset, highest_signal_count, mapped_capture)
    
    # Save the DataFrame to a CSV file in the specified output directory
    output_file = os.path.join(args.output_dir, f"output_{os.path.basename(args.log_filepath).split('.')[0]}.csv")
    save_dataframe_to_csv(df_benign, args.window_size, args.offset, output_file)

    # Save the combined embeddings to disk
    embeddings_output_file = os.path.join(args.output_dir, "combined_node_embeddings.pkl")
    with open(embeddings_output_file, "wb") as f:
        pickle.dump(all_combined_node_embeddings, f)

if __name__ == "__main__":
    main()
