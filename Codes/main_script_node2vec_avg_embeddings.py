#!/usr/bin/env python
# coding: utf-8

# In[10]:


get_ipython().system('pip install setuptools')


# In[11]:


cd C:\Users\willi\actt\src


# In[12]:


get_ipython().system('python setup.py develop')


# In[13]:


get_ipython().system('pip install bitstring')
get_ipython().system('pip install seaborn')
get_ipython().system('pip install cantools')
import os 
import sys
import numpy as np
from collections import defaultdict
import CAN_objects.aid_message
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from generalFunctions import unpickle
import subprocess

import importlib
importlib.reload(CAN_objects.aid_message)


# In[14]:


import os
import sys

sys.path.insert(0, os.path.dirname(os.getcwd()) + "/code/")
get_ipython().system('pip install opencv-python')
get_ipython().system('pip install helper_functions')
import helper_functions
get_ipython().system('pip install networkx')
import networkx

get_ipython().system('pip install tqdm')
from tqdm import tqdm

import numpy as np
import pandas as pd
from scipy.integrate import quad

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import json
import fnmatch
import scipy

get_ipython().system('pip install scikit-learn')
from sklearn.metrics import auc
from sklearn.covariance import EllipticEnvelope


# In[15]:


# path for log file 'C:\Users\willi\OneDrive\Desktop\Research\oak_ridge_in_vehicle\road'
file = 'C:\\Users\\willi\\OneDrive\\Desktop\\Research\\oak_ridge_in_vehicle\\road\\ambient\\ambient_dyno_drive_basic_long.log'


# ### Calling out make_can_df

# In[16]:


def make_can_df(log_filepath):

    can_df = pd.read_fwf(
        log_filepath, delimiter = ' '+ '#' + '('+')',
        skiprows = 1,skipfooter=1,
        usecols = [0,2,3],
        dtype = {0:'float64', 1:str, 2: str},
        names = ['time','pid', 'data'] )

    can_df.pid = can_df.pid.apply(lambda x: int(x,16))
    can_df.data = can_df.data.apply(lambda x: x.zfill(16)) #pad with 0s on the left for data with dlc < 8
    can_df.time = can_df.time - can_df.time.min()
    return can_df[can_df.pid<=0x700]


# In[17]:


# Read the log file and parse the contents into a DataFrame
df = make_can_df(file)
df


# In[18]:


# Print the first few rows and the column names of the DataFrame
df = make_can_df(file)
print(df.head())
print(df.columns)


# In[19]:


# Sort the DataFrame by the 'time' column and round the 'time' values to 2 decimal places
df_sorted = df.sort_values('time')
df_sorted['time'] = df_sorted['time'].round(2)


# In[20]:


# Divide the sorted DataFrame into time slices of duration 10.0 and print each slice along with its label
time_slice_duration = 10.0
num_slices = int(df_sorted['time'].max() / time_slice_duration) + 1
for i in range(num_slices):
    start_time = i * time_slice_duration
    end_time = (i + 1) * time_slice_duration
    time_slice_label = f"Time Slice {i + 1}: ({start_time}, {end_time}]"
    time_slice_df = df_sorted[(df_sorted['time'] > start_time) & (df_sorted['time'] <= end_time)]
    print(f"{time_slice_label}\n{time_slice_df}\n")


# ### Building graphs for each time window (edges)

# ##### Save the Graphs to Disk:

# In[21]:


import pickle
import networkx as nx
import pandas as pd
from tqdm import tqdm


# Define the time slice duration in seconds
time_slice_duration = 10.0

# Determine the number of time slices
num_slices = int(df['time'].max() / time_slice_duration) + 1

# Lists to store the graphs
graphs = []

# Using tqdm in the loop for progress tracking
for i in tqdm(range(num_slices), desc="Generating Graphs"):
    start_time = i * time_slice_duration
    end_time = (i + 1) * time_slice_duration
    time_slice_df = df[(df['time'] > start_time) & (df['time'] <= end_time)]

    # Build the graph for the current time slice
    G = nx.DiGraph()
    node_ids = time_slice_df['pid'].unique().tolist()
    G.add_nodes_from(node_ids)

    for j in range(len(time_slice_df) - 1):
        source = time_slice_df.iloc[j]['pid']
        target = time_slice_df.iloc[j+1]['pid']

        if G.has_edge(source, target):
            G[source][target]['weight'] += 1
        else:
            G.add_edge(source, target, weight=1)

    # Append the graph to the list
    graphs.append(G)

# Save the graphs to a pickle file
with open("saved_graphs.pkl", "wb") as f:
    pickle.dump(graphs, f)


# #### Load the Graphs from Disk and Plot:

# In[22]:


import pickle
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

# Load the graphs from the pickle file
with open("saved_graphs.pkl", "rb") as f:
    loaded_graphs = pickle.load(f)

# Using tqdm in the loop for progress tracking
for i, G in tqdm(enumerate(loaded_graphs), total=len(loaded_graphs), desc="Plotting Graphs"):
    time_slice_label = f"Time Slice {i + 1}"
    plt.figure(i)
    nx.draw(G, with_labels=True, node_color='lightblue', node_size=500, font_size=10, edge_color='gray')
    plt.title(time_slice_label)
    plt.text(0.05, 0.95, f'Nodes: {len(G.nodes())}\nEdges: {len(G.edges())}', transform=plt.gca().transAxes, verticalalignment='top')

plt.show()


# In[23]:


# Import necessary modules and classes for working with CAN (Controller Area Network) captures
import os
from CAN_objects.capture import MappedCapture, MatchedCapture


# In[24]:


# Define the file path for the CAN capture and load the data from the pickle file
cancap_filepath = os.path.join("C:\\Users\\willi\\Dropbox\\data-cancaptures", "road_ambient_dyno_drive_basic_long_050305_002000", "capture.pkl")
cancap = unpickle(cancap_filepath)


# In[25]:


# Define the file path for the ground truth DBC file and create a MappedCapture object from the CAN capture data and DBC file
ground_truth_dbc_fpath = os.path.join("C:\\Users\\willi\\Dropbox\\data-cancaptures\\DBC", "anonymized_020822_030640.dbc")
mapped_capture = MappedCapture.init_from_dbc(cancap, ground_truth_dbc_fpath)


# ### Function to extract the mean and standard deviation for a specific ID in a given time window.

# In[26]:


def extract_mean_std_for_id(node_id, start_time, end_time, mapped_capture):
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


# ### Annotate the nodes with the mean and std attributes

# ##### Generate and Save Figures

# In[27]:


get_ipython().system('pip install plotly')
import plotly.graph_objects as go
import networkx as nx
import pandas as pd
import math
import pickle
from tqdm import tqdm

# Function to create annotated edges with arrows
def create_annotated_edges(G, pos):
    edge_traces = []
    arrow_annotations = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace = go.Scatter(x=[x0, x1, None], y=[y0, y1, None],
                                line=dict(width=0.1, color='gray'),
                                mode='lines')
        edge_traces.append(edge_trace)

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

# Create an empty list to store figures
all_figures = []

# Loop through time slices and create directed graphs
for i in tqdm(range(num_slices), desc="Generating Figures"):
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

    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    node_text = [f"Node ID: {node}<br>" + "<br>".join([f"{k}: {v}" for k, v in attrs.items()]) for node, attrs in G.nodes(data=True)]

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
    
    # Store the figures
    all_figures.append(fig)

# Save the figures to disk
with open("saved_figures.pkl", "wb") as f:
    pickle.dump(all_figures, f)


# ##### Loading Graphs from the Pickle File 

# In[28]:


os.getcwd()


# In[29]:


import pickle
from tqdm.notebook import tqdm

# Load the saved figures from disk
with open("saved_figures.pkl", "rb") as f:
    loaded_figures = pickle.load(f)

# Display the first 10 graphs
for fig in tqdm(loaded_figures[:10], desc="Displaying Figures"):
    fig.show()

# To display a specific graph later, we can do:
# loaded_figures[index].show()
# For example, to show the first graph, use:
# loaded_figures[0].show()


# ### display the graph for (time slice 126):

# In[30]:


all_figures[125].show()


# ### Load the saved graphs from the disk:

# In[31]:


print("Loading graphs from disk...")
with open("saved_graphs.pkl", "rb") as f:
    all_graphs_original = pickle.load(f)  # Keeping original indexing


# ### Node2vec implementation

# In[45]:


import multiprocessing
num_cores = multiprocessing.cpu_count()
print(num_cores)


# In[46]:


from node2vec import Node2Vec
from tqdm import tqdm
import numpy as np

# Initialize a list to hold embeddings for all nodes across all graphs
all_node_embeddings = []

# Iterate over all graphs
for G in tqdm(all_graphs_original, desc="Generating Node Embeddings"):
    # Embed the graph using node2vec; p and q values can be adjusted as needed
    # since our graph is weighted, Node2Vec automatically considers the weights
    node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=30, p=2, q=0.5)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    
    node_embeddings = {}
    for node in G.nodes():
        # Check if node was embedded
        if node in model.wv:
            node_emb = model.wv[node]
            node_embeddings[node] = node_emb
        else:
            # Assign zero vector for nodes that didn't get an embedding
            node_embeddings[node] = np.zeros(model.vector_size)
    
    all_node_embeddings.append(node_embeddings)


# ### Combining Node Embeddings with Node Attributes

# In[48]:


all_combined_node_embeddings = []

# Calculate the maximum number of attributes across all nodes and all graphs
max_attributes = max([len(extract_mean_std_for_id(node, start_time, end_time, mapped_capture))
                      for graph in all_graphs_original
                      for node in graph.nodes()])

# Iterate over all graphs and their index
for idx, G in tqdm(enumerate(all_graphs_original), desc="Combining Node Embeddings with Attributes"):
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


# In[50]:


# Display the first few combined embeddings for the first graph
print(all_combined_node_embeddings[0])


# In[51]:


all_combined_node_embeddings[0][1694].shape


# In[52]:


# Get combined embeddings for the first graph
first_graph_embeddings = all_combined_node_embeddings[0]

# Select an arbitrary node's embedding from the first graph
some_node_id = list(first_graph_embeddings.keys())[0]
node_embedding = first_graph_embeddings[some_node_id]

# Print the dimensions (length) of the node's embedding
print("Dimensions of the combined embedding for a node in the first graph:", len(node_embedding))


# In[53]:


for j in all_combined_node_embeddings[0].keys():
    print (all_combined_node_embeddings[0][j].shape)
    


# In[54]:


print(len(all_combined_node_embeddings))
print(len(all_combined_node_embeddings[0]))


# This indicates that the first graph out of those 126 has embeddings for 105 nodes.

# In[55]:


for idx, G in enumerate(all_graphs_original):
    # Get all nodes from the graph
    original_nodes = set(G.nodes())
    
    # Get all nodes from the embeddings
    embedded_nodes = set(all_combined_node_embeddings[idx].keys())
    
    # Check if the nodes from the graph match the nodes in the embeddings
    assert original_nodes == embedded_nodes, f"Mismatch in graph {idx}."


# Every graph in all_graphs_original has a corresponding entry in all_combined_node_embedding.
# And every node in each graph of all_graphs_original has a corresponding embedding in 
# the associated entry of all_combined_node_embeddings.Hence our embeddings are correctly aligned with the nodes of the original graphs. 

# In[56]:


from sklearn.cluster import KMeans

# Extract combined embeddings for the chosen graph (assuming graph_idx=0)
graph_idx = 0
embeddings = all_combined_node_embeddings[graph_idx]

# Extract node ids and their embeddings
node_ids, node_embeddings = zip(*embeddings.items())
node_embeddings = np.array(node_embeddings)

# Cluster the embeddings into 5 clusters using KMeans
kmeans = KMeans(n_clusters=5, random_state=0).fit(node_embeddings)
node_labels = kmeans.labels_


# In[57]:


import matplotlib.colors
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Define distinct colors for each cluster
colors = ["red", "blue", "green", "yellow", "purple"]

# Apply t-SNE on the embeddings
tsne = TSNE(n_components=2, random_state=42)
transformed_embeddings = tsne.fit_transform(node_embeddings)

plt.figure(figsize=(12, 10))
scatter = plt.scatter(
    transformed_embeddings[:, 0],
    transformed_embeddings[:, 1],
    c=node_labels,
    cmap=matplotlib.colors.ListedColormap(colors),
    alpha=0.7,
    edgecolor="w",
    s=100  # Increased size for nodes
)
cbar = plt.colorbar(scatter, ticks=np.arange(5))
cbar.set_label('Cluster ID')
plt.title("t-SNE Visualization of Combined Node Embeddings with Cluster Coloring")
plt.show()


# In[58]:


plt.figure(figsize=(12, 10))
plt.axes().set(aspect="equal")
plt.scatter(
    transformed_embeddings[:, 0],
    transformed_embeddings[:, 1],
    c=node_labels,
    cmap=matplotlib.colors.ListedColormap(colors),
    alpha=0.7,
    edgecolor="w",
    s=100  # Increased size for nodes
)
for i, node_id in enumerate(node_ids):
    plt.annotate(str(node_id), (transformed_embeddings[i, 0], transformed_embeddings[i, 1]))
plt.title("t-SNE Visualization of Combined Node Embeddings with KMeans Clusters")
plt.colorbar().set_label('Cluster ID')
plt.show()


# In[59]:


##To ensure that the graph embeddings consider the weights, directions

for u, v, data in G.edges(data=True):
    print(u, v, data)
##


#  1255: This is the source node of the edge.
#     339: This is the target node of the edge.
#     {'weight': 10}: This is a dictionary containing edge attributes. In this case, the only attribute shown is weight, which represents the weight of the edge from node 1255 to node 339. The weight here is 10.
# 
# So, this line essentially means there's a directed edge from node 1255 to node 339 with a weight of 10.

#  ### A single embedding for the 1st graph by taking the average of the concatenated node embeddings:

# In[60]:


import numpy as np

# Initialize an empty list to store concatenated embeddings
concatenated_embeddings = []

# Select a graph index to process
graph_idx = 0

# Iterate through all nodes in the selected graph
for node in all_combined_node_embeddings[graph_idx]:
    # Get the concatenated embedding for the current node
    concatenated_embedding = all_combined_node_embeddings[graph_idx][node]
    
    # Append the concatenated embedding to the list
    concatenated_embeddings.append(concatenated_embedding)

# Calculate the average of the concatenated embeddings
average_embedding = np.mean(concatenated_embeddings, axis=0)

# Now, 'average_embedding' contains a single vector representing the entire graph.
# You can use this vector for further analysis or machine learning tasks.


# In[61]:


average_embedding


# #### Repeating the process for every graph in the database

# In[70]:


import numpy as np

# Initialize an empty list to store average embeddings for all graphs
all_average_embeddings = []

# Iterate through all graphs
for graph_idx in range(len(all_combined_node_embeddings)):
    # Initialize an empty list to store concatenated embeddings
    concatenated_embeddings = []

    # Iterate through all nodes in the selected graph
    for node in all_combined_node_embeddings[graph_idx]:
        # Get the concatenated embedding for the current node
        concatenated_embedding = all_combined_node_embeddings[graph_idx][node]
        
        # Replace NaN values with zeros
        concatenated_embedding = np.nan_to_num(concatenated_embedding, nan=0.0)

        # Append the concatenated embedding to the list
        concatenated_embeddings.append(concatenated_embedding)

    # Calculate the average of the concatenated embeddings for the current graph
    average_embedding = np.mean(concatenated_embeddings, axis=0)

    # Append the average embedding to the list of average embeddings for all graphs
    all_average_embeddings.append(average_embedding)

# Now, 'all_average_embeddings' contains a list of average embeddings, one for each graph.


# In[71]:


all_average_embeddings


# In[72]:


import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Combine all average embeddings into a single numpy array
average_embeddings_array = np.array(all_average_embeddings)

# Apply t-SNE on the embeddings
tsne = TSNE(n_components=2, random_state=42)
transformed_embeddings = tsne.fit_transform(average_embeddings_array)

# Plot the t-SNE representation
plt.figure(figsize=(12, 10))
plt.scatter(transformed_embeddings[:, 0], transformed_embeddings[:, 1])

# Annotate the points with graph indices
for graph_idx in range(len(all_average_embeddings)):
    plt.annotate(str(graph_idx), (transformed_embeddings[graph_idx, 0], transformed_embeddings[graph_idx, 1]))

plt.title("t-SNE Visualization of Average Graph Embeddings")
plt.show()


# In[73]:





# In[ ]:





# In[ ]:





# In[ ]:




