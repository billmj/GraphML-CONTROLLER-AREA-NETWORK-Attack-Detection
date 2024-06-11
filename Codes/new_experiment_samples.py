# new_experiment_samples.py

import os
import argparse
import subprocess
import shutil
import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
from node2vec import Node2Vec
import pickle
from generalFunctions import unpickle
from CAN_objects.capture import MappedCapture
from math import ceil

# Import your custom modules here
from data_processing import make_can_df, generate_graphs_from_data_samples, process_dataframe_samples
from embedding_generation import (
    generate_node_embeddings, combine_embeddings_with_attributes,
    compute_average_embeddings, create_dataframe_from_embeddings, display_combined_embeddings
)
from utils import load_mapped_capture, find_node_with_highest_signals, save_dataframe_to_csv

# Predefined dictionary for benign log files and their corresponding pickle folders
BENIGN_LOG_TO_PKL = {
    "ambient_dyno_drive_basic_long.log": "road_ambient_dyno_drive_basic_long_050305_002000",
    "ambient_dyno_drive_basic_short.log": "road_ambient_dyno_drive_basic_short_020822_030640",
    "ambient_dyno_drive_benign_anomaly.log": "road_ambient_dyno_drive_benign_anomaly_030804_082640",
    "ambient_dyno_drive_extended_long.log": "road_ambient_dyno_drive_extended_long_040716_134640",
    "ambient_dyno_drive_extended_short.log": "road_ambient_dyno_drive_extended_short_021215_195320",
    "ambient_dyno_drive_radio_infotainment.log": "road_ambient_dyno_drive_radio_infotainment_041109_063320",
    "ambient_dyno_drive_winter.log": "road_ambient_dyno_drive_winter_030410_144000",
    "ambient_dyno_exercise_all_bits.log": "road_ambient_dyno_exercise_all_bits_030410_144000",
    "ambient_dyno_idle_radio_infotainment.log": "road_ambient_dyno_idle_radio_infotainment_030410_144000",
    "ambient_dyno_reverse.log": "road_ambient_dyno_reverse_040322_190000"
}

# Predefined dictionary for attack log files and their corresponding pickle folders with attack intervals
ATTACK_LOG_TO_PKL = {
    "correlated_signal_attack_1_masquerade.log": ("road_attack_correlated_signal_attack_1_masquerade_030804_082640", (9.191851, 30.050109)),
    "correlated_signal_attack_2_masquerade.log": ("road_attack_correlated_signal_attack_2_masquerade_031128_011320", (6.830477, 28.225908)),
    "correlated_signal_attack_3_masquerade.log": ("road_attack_correlated_signal_attack_3_masquerade_040322_190000", (4.318482, 16.95706)),
    "max_speedometer_attack_1_masquerade.log": ("road_attack_max_speedometer_attack_1_masquerade_060215_054000", (42.009204, 66.449011)),
    "max_speedometer_attack_2_masquerade.log": ("road_attack_max_speedometer_attack_2_masquerade_060611_002640", (16.009225, 47.408246)),
    "max_speedometer_attack_3_masquerade.log": ("road_attack_max_speedometer_attack_3_masquerade_061004_181320", (9.516489, 70.587285)),
    "reverse_light_off_attack_1_masquerade.log": ("road_attack_reverse_light_off_attack_1_masquerade_080110_162000", (16.627923, 23.347311)),
    "reverse_light_off_attack_2_masquerade.log": ("road_attack_reverse_light_off_attack_2_masquerade_080505_110640", (13.168608, 36.87663)),
    "reverse_light_off_attack_3_masquerade.log": ("road_attack_reverse_light_off_attack_3_masquerade_080829_045320", (16.524085, 40.862015)),
    "reverse_light_on_attack_1_masquerade.log": ("road_attack_reverse_light_on_attack_1_masquerade_091205_030000", (18.929177, 38.836015)),
    "reverse_light_on_attack_2_masquerade.log": ("road_attack_reverse_light_on_attack_2_masquerade_100330_214640", (20.407134, 57.297253)),
    "reverse_light_on_attack_3_masquerade.log": ("road_attack_reverse_light_on_attack_3_masquerade_100724_153320", (23.070278, 46.580686)),
    "max_engine_coolant_temp_attack_masquerade.log": ("road_attack_max_engine_coolant_temp_attack_masquerade_041109_063320", (19.979078, 24.170183))
}

def run_experiment(log_filepath, window_size, offset, pkl_folder, output_dir, attack_type, attack_interval=None):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read the CAN log and process it
    df = make_can_df(log_filepath)
    process_dataframe_samples(df, window_size, offset)

    # Generate graphs from data
    all_graphs = generate_graphs_from_data_samples(df, window_size, offset)

    # Generate node embeddings
    all_node_embeddings = generate_node_embeddings(all_graphs)

    # Load mapped_capture for attribute extraction using the specified pkl folder
    mapped_capture = load_mapped_capture(pkl_folder)

    # Combine the embeddings with attributes
    all_combined_node_embeddings = combine_embeddings_with_attributes(all_graphs, all_node_embeddings, window_size, mapped_capture)

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

    # Create the DataFrame from embeddings
    df_benign = create_dataframe_from_embeddings(concatenated_embeddings, log_filepath, offset, highest_signal_count, mapped_capture)

    # Label data if it is an attack log
    if attack_interval:
        start_attack, end_attack = attack_interval
        df_benign['Label'] = df_benign['Timestamp'].apply(lambda x: 1 if start_attack <= x <= end_attack else 0)
    else:
        df_benign['Label'] = 0  # For benign logs, label everything as 0

    # Display the first few rows of the DataFrame
    print(df_benign.head())

    # Display the last few rows of the DataFrame
    print(df_benign.iloc[-5:])

    # Save the DataFrame to a CSV file
    output_csv_path = os.path.join(output_dir, f"{attack_type}_w{window_size}_o{offset}.csv")
    save_dataframe_to_csv(df_benign, window_size, offset, log_filepath)
    
    # Save the DataFrame to the specific attack's folder as well
    specific_attack_output_dir = os.path.join(output_dir, attack_type, f"w{window_size}_o{offset}")
    if not os.path.exists(specific_attack_output_dir):
        os.makedirs(specific_attack_output_dir)
    df_benign.to_csv(os.path.join(specific_attack_output_dir, f"{attack_type}_w{window_size}_o{offset}.csv"), index=False)

    # Save the combined embeddings to disk
    with open(os.path.join(output_dir, "combined_node_embeddings.pkl"), "wb") as f:
        pickle.dump(all_combined_node_embeddings, f)

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Run new CAN log experiment based on sample count.')
    parser.add_argument('--window-size', type=int, required=True, help='Number of samples in each window.')
    parser.add_argument('--offset', type=int, required=True, help='Number of samples to shift for each window.')
    parser.add_argument('--output-dir', type=str, default=None, help='Directory to save processed files. If not provided, the default output directory will be used.')
    parser.add_argument('--attack-type', type=str, choices=['benign', 'attack'], required=True, help='Specify whether to process benign or attack logs.')

    args = parser.parse_args()

        # Ensure the output directory exists
    if args.output_dir is None:
        default_output_dir = "C:\\Users\\willi\\CAN_experiments\\New_Experiments"
        args.output_dir = default_output_dir

    # Determine log files and directories based on attack type
    if args.attack_type == 'benign':
        log_directory = "C:\\Users\\willi\\OneDrive\\Desktop\\Research\\oak_ridge_in_vehicle\\road\\ambient\\"
        log_to_pkl = BENIGN_LOG_TO_PKL
    else:
        log_directory = "C:\\Users\\willi\\OneDrive\\Desktop\\Research\\oak_ridge_in_vehicle\\road\\attacks\\"
        log_to_pkl = ATTACK_LOG_TO_PKL

    # Loop through each log file and run the experiment
    for log_file, pkl_folder_info in log_to_pkl.items():
        log_filepath = os.path.join(log_directory, log_file)
        
        # Determine if it is an attack file with an interval
        if args.attack_type == 'attack':
            pkl_folder, attack_interval = pkl_folder_info
        else:
            pkl_folder = pkl_folder_info
            attack_interval = None
        
        # Printing the name of the dataset being processed
        print(f"Processing {args.attack_type} log file: {log_file}...")

        # Run the experiment
        run_experiment(log_filepath, args.window_size, args.offset, pkl_folder, args.output_dir, args.attack_type, attack_interval)

if __name__ == "__main__":
    main()

