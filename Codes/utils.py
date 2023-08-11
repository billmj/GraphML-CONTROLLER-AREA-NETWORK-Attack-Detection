# utils.py

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import bitstring
import cantools
import cv2
import helper_functions
from tqdm import tqdm
from sklearn import some_module
import plotly
from scipy.integrate import quad
from sklearn.covariance import EllipticEnvelope

# Function to create annotated edges with arrows
def create_annotated_edges(G, pos):
    edge_traces = []
    arrow_annotations = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        # Create line trace for edge
        edge_trace = go.Scatter(x=[x0, x1, None], y=[y0, y1, None],
                                line=dict(width=0.1, color='gray'),
                                mode='lines')
        edge_traces.append(edge_trace)

        # Create arrow annotation for the directed edge
        arrow_annotation = dict(
            ax=x0,
            ay=y0,
            axref='x',
            ayref='y',
            x=x1,
            y=y1,
            xref='x',
            yref='y',
            showarrow=True,
            arrowhead=4,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor='gray'
        )
        arrow_annotations.append(arrow_annotation)

    return edge_traces, arrow_annotations

# Function to extract the mean and standard deviation for a specific ID in a given time window
def extract_mean_std_for_id(node_id, start_time, end_time, mapped_capture):
    mp = mapped_capture.mapped_payload_dict[node_id]
    mask = (mp.times >= start_time) & (mp.times < end_time)
    sub_df = pd.DataFrame()

    for signal in mp.signal_list:
        signal_values = signal.values[mask]
        interpolated_values = pd.Series(signal_values).interpolate(method='linear', limit_area='inside').values
        sub_df[signal.name] = interpolated_values

    mean_std_dict = {}
    for col, (mean, std) in zip(sub_df.columns, zip(sub_df.mean().values, sub_df.std().values)):
        mean_std_dict[f"{col}_mean"] = round(mean, 2)
        mean_std_dict[f"{col}_std"] = round(std, 2)

    return mean_std_dict

# Function to read CAN data log file and create DataFrame
def make_can_df(log_filepath):
    can_df = pd.read_fwf(
        log_filepath, delimiter = ' '+ '#' + '('+')',
        skiprows = 1, skipfooter = 1,
        usecols = [0, 2, 3],
        dtype = {0: 'float64', 1: str, 2: str},
        names = ['time', 'pid', 'data'])
    
    can_df.pid = can_df.pid.apply(lambda x: int(x, 16))
    can_df.data = can_df.data.apply(lambda x: x.zfill(16))
    can_df.time = can_df.time - can_df.time.min()
    
    return can_df[can_df.pid <= 0x700]

# Define time slice duration
time_slice_duration = 10.0

# Loop through time slices and create directed graphs
def create_directed_graphs(df, mapped_capture):
    for i in range(num_slices):
        start_time = i * time_slice_duration
        end_time = (i + 1) * time_slice_duration
        time_slice_label = f"Time Slice {i + 1}: ({start_time}, {end_time}]"
        time_slice_df = df[(df['time'] > start_time) & (df['time'] <= end_time)]

        G = nx.DiGraph()
        node_ids = time_slice_df['pid'].unique().tolist()
        G.add_nodes_from(node_ids)

        for node_id in node_ids:
            mean_std_dict = extract_mean_std_for_id(node_id, start_time, end_time, mapped_capture)
            nx.set_node_attributes(G, {node_id: mean_std_dict})

        for j in range(len(time_slice_df) - 1):
            source = time_slice_df.iloc[j]['pid']
            target = time_slice_df.iloc[j + 1]['pid']

            if G.has_edge(source, target):
                G[source][target]['weight'] += 1
            else:
                G.add_edge(source, target, weight=1)

        pos = nx.spring_layout(G, k=0.15)
        edge_traces, arrow_annotations = create_annotated_edges(G, pos)

        fig = go.Figure(data=[*edge_traces, node_trace],
                        layout=go.Layout(
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=0, l=0
