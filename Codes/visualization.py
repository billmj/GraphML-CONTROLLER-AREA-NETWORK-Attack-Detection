# visualization.py

# Import necessary libraries
import plotly.graph_objects as go
import networkx as nx
from IPython.display import clear_output
import pandas as pd
import math

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

# Define time slice duration
time_slice_duration = 10.0
num_slices = int(df['time'].max() / time_slice_duration) + 1

# Loop through time slices and create directed graphs
for i in range(num_slices):
    start_time = i * time_slice_duration
    end_time = (i + 1) * time_slice_duration
    time_slice_label = f"Time Slice {i + 1}: ({start_time}, {end_time}]"
    time_slice_df = df[(df['time'] > start_time) & (df['time'] <= end_time)]

    # Create directed graph
    G = nx.DiGraph()
    node_ids = time_slice_df['pid'].unique().tolist()
    G.add_nodes_from(node_ids)

    # Add node attributes
    for node_id in node_ids:
        mean_std_dict = extract_mean_std_for_id(node_id, start_time, end_time, mapped_capture)
        nx.set_node_attributes(G, {node_id: mean_std_dict})

    # Add edges with weights
    for j in range(len(time_slice_df) - 1):
        source = time_slice_df.iloc[j]['pid']
        target = time_slice_df.iloc[j + 1]['pid']
        if G.has_edge(source, target):
            G[source][target]['weight'] += 1
        else:
            G.add_edge(source, target, weight=1)

    # Layout, annotations and visualizations
    pos = nx.spring_layout(G, k=0.15)
    edge_traces, arrow_annotations = create_annotated_edges(G, pos)

    # Create node positions and texts
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    node_text = [f"Node ID: {node}<br>" + "<br>".join([f"{k}: {v}" for k, v in attrs.items()]) for node, attrs in G.nodes(data=True)]

    # Create node trace for visualization
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=[str(node) for node in G.nodes()],
        textposition='middle center',
        textfont=dict(size=10, color='black'),
        hoverinfo='text',
        hovertext=node_text,
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=30,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2
        )
    )

    # Build and display the figure
    fig = go.Figure(data=[*edge_traces, node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=40),
                        title=dict(
                            text=time_slice_label,
                            x=0.5,
                            xanchor='center',
                            font=dict(
                                size=14,
                                color='black'
                            )
                        ),
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False),
                        annotations=arrow_annotations
                    )
    )
    fig.show()

    # Prompt to continue to next plot
    input("Press Enter to continue...")
    clear_output(wait=True)
