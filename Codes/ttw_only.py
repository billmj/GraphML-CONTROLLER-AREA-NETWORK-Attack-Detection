# ttw_only.py
from imports import *
from data_processing import make_can_df, process_dataframe
from ttw_calculations import calculate_ttw, save_ttw_results

def main():
    parser = argparse.ArgumentParser(description='Calculate TTW for CAN log data.')
    parser.add_argument('--window-size', type=float, required=True, help='Size of the window for slicing data.')
    parser.add_argument('--offset', type=float, required=True, help='Offset for slicing data.')
    parser.add_argument('--pkl-folder', type=str, required=True, help='Folder name for the .pkl file.')
    parser.add_argument('log_filepath', type=str, help='Path to the CAN log file.')
    args = parser.parse_args()

    # Read the CAN log and process it
    df = make_can_df(args.log_filepath)
    process_dataframe(df, args.window_size, args.offset)

    # Generate graphs and calculate TTW
    all_graphs, ttws, avg_ttw = calculate_ttw(df, args.window_size, args.offset)

    # Save TTW results to a CSV file
    save_ttw_results(ttws, avg_ttw, args.log_filepath, args.window_size, args.offset)

    print("TTW calculation complete. Embedding generation skipped.")

if __name__ == "__main__":
    main()