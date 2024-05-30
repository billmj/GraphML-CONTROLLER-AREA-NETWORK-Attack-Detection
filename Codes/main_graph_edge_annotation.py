import os
import networkx as nx
import matplotlib.pyplot as plt
from utils import make_can_df, create_time_slices, build_graphs

# Define the log file path
log_filepath = 'C:\\Users\\willi\\OneDrive\\Desktop\\Research\\oak_ridge_in_vehicle\\road\\ambient\\ambient_dyno_drive_basic_long.log'

# Define the time slice duration in seconds (parametrize it as needed)
time_slice_duration = 10.0

def main():
    # Read the log file and parse the contents into a DataFrame
    df = make_can_df(log_filepath)

    # Create time slices
    time_slices = create_time_slices(df, time_slice_duration)

    # Build graphs for each time window
    nodes_counts, edges_counts, graphs = build_graphs(time_slices)

    # Output results as needed
    for label, G in graphs:
        print(label)
        print(f'Nodes: {len(G.nodes())}')
        print(f'Edges: {len(G.edges())}')
        plt.figure()
        nx.draw(G, with_labels=True, node_color='lightblue', node_size=500, font_size=10, edge_color='gray')
        plt.title(label)
        plt.show()

if __name__ == "__main__":
    main()
