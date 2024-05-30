#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install setuptools')


# In[2]:


cd C:\Users\wm6\actt\src


# In[3]:


get_ipython().system('python setup.py develop')


# In[4]:


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


# In[5]:


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


# In[6]:


# path for log file 'C:\Users\wm6\Dropbox\cant2can-ldrd\can-data\open_dataset_logs\road\ambient'
file = 'C:\\Users\\wm6\\Dropbox\\cant2can-ldrd\\can-data\\open_dataset_logs\\road\\ambient\\ambient_dyno_drive_basic_long.log'


# ### Calling out make_can_df

# In[7]:


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


# In[8]:


# Read the log file and parse the contents into a DataFrame
df = make_can_df(file)
df


# In[9]:


# Sort the DataFrame based on the 'time' column
df_sorted = df.sort_values('time')

# Format the 'time' values to two decimal points
df_sorted['time'] = df_sorted['time'].round(2)

# Print the new sorted DataFrame
print(df_sorted)


# In[10]:


# Define the time slice duration in seconds
time_slice_duration = 10.0

# Determine the number of time slices
num_slices = int(df_sorted['time'].max() / time_slice_duration) + 1

# Iterate over the time slices
for i in range(num_slices):
    start_time = i * time_slice_duration
    end_time = (i + 1) * time_slice_duration
    time_slice_label = f"Time Slice {i + 1}: ({start_time}, {end_time}]"
    time_slice_df = df_sorted[(df_sorted['time'] > start_time) & (df_sorted['time'] <= end_time)]
    print(f"{time_slice_label}\n{time_slice_df}\n")


# ### Building graphs for each time window

# In[11]:


import networkx as nx
import matplotlib.pyplot as plt

# Define the time slice duration in seconds
time_slice_duration = 10.0

# Determine the number of time slices
num_slices = int(df['time'].max() / time_slice_duration) + 1

# Iterate over the time slices
for i in range(num_slices):
    start_time = i * time_slice_duration
    end_time = (i + 1) * time_slice_duration
    time_slice_label = f"Time Slice {i + 1}: ({start_time}, {end_time}]"
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

    # Plot the graph
    plt.figure(i)
    nx.draw(G, with_labels=True, node_color='lightblue', node_size=500, font_size=10, edge_color='gray')
    plt.title(time_slice_label)

plt.show()


# In[ ]:





# In[ ]:





# In[12]:


import os
from CAN_objects.capture import MappedCapture, MatchedCapture


# In[13]:


cancap_filepath = os.path.join("C:\\Users\\wm6\\Dropbox\\data-cancaptures", "road_ambient_dyno_drive_basic_long_050305_002000", "capture.pkl")
cancap = unpickle(cancap_filepath)


# In[14]:


ground_truth_dbc_fpath = os.path.join("C:\\Users\\wm6\\Dropbox\\data-cancaptures\\DBC", "anonymized_020822_030640.dbc")
mapped_capture = MappedCapture.init_from_dbc(cancap, ground_truth_dbc_fpath)


# In[15]:


dir(mapped_capture)


# In[16]:


mapped_capture.has_diagnostics()


# In[17]:


mapped_capture.get_elapsed_sec()


# In[18]:


mapped_capture


# In[19]:


mapped_capture.mapped_payload_dict


# In[20]:


mapped_capture.mapped_payload_dict.keys()


# In[21]:


dir(mapped_capture)


# ### Explore the Output for the IDS

# In[22]:


mp = mapped_capture.mapped_payload_dict[1760]


# In[23]:


dir(mp)


# ### Time series plot

# In[24]:


mapped_capture.get_elapsed_sec()


# In[25]:


mp.payload_arr.shape


# In[26]:


mp.get_transmission_rate()


# In[27]:


mp.sending_ECU


# In[28]:


mp.plot_signals()


# In[29]:


mp.payload_arr


# In[30]:


mp.payload_arr.shape


# In[31]:


mp.payload_arr.T[:, 910:920]


# In[32]:


display(mp.signal_list)
print(type(mp.signal_list[0]), len(mp.signal_list))


# In[33]:


s = mp.signal_list[0]
s


# In[34]:


type(s.plot())


# In[35]:


dir(s)


# In[36]:


print(min(s.times), max(s.times))


# In[37]:


s.values


# In[38]:


s.unit


# In[39]:


s= mp.signal_list[1]
s


# In[40]:


type(s.plot())


# In[41]:


dir(s)


# In[42]:


print(min(s.times), max(s.times))


# In[43]:


s.values


# In[44]:


s.unit


# In[45]:


s = mp.signal_list[2]
s


# In[46]:


type(s.plot())


# In[47]:


dir(s)


# In[48]:


print(min(s.times), max(s.times))


# In[49]:


s.values


# In[ ]:





# In[50]:


s = mp.signal_list[3]
s


# In[51]:


type(s.plot())


# In[52]:


dir(s)


# In[53]:


print(min(s.times), max(s.times))


# In[54]:


s.values


# In[ ]:





# In[55]:


import pandas as pd

# Initialize lists to store the extracted data
signal_names = []
data_dict = {}

# Define the time windows (every 10 seconds)
time_windows = range(0, int(mp.times[-1]), 10)

# Loop through the signal_list and extract information for each time window
for window_start in time_windows:
    window_end = window_start + 10
    
    # Create a sub-DataFrame for the current time window
    sub_df = pd.DataFrame(index=range(10))  # Assuming 10 samples in each time window
    
    for signal in mp.signal_list:
        # Get the indices for the current time window
        mask = (signal.times >= window_start) & (signal.times < window_end)
        
        # Get the values within the current time window and interpolate for consistent time resolution
        signal_values = signal.values[mask]
        interpolated_values = pd.Series(signal_values).interpolate().values
        
        # Add the interpolated signal values to the sub-DataFrame
        sub_df[signal.name] = interpolated_values[:10]  # Truncate to 10 samples if needed
    
    # Append the sub-DataFrame to the data dictionary with the window start as the key
    data_dict[f"{window_start} - {window_end}"] = sub_df

# Create a pandas DataFrame with the extracted data
df = pd.concat(data_dict, axis=1)

# Display the DataFrame
print(df)


# In[ ]:





# ### mean and sd for each signal per time window [mean,sd] for ID 1760

# In[56]:


import pandas as pd

# Initialize lists to store the extracted data
signal_names = []
data_dict = {}

# Initialize a list to store the summary data for each time window
summary_data = []

# Define the time windows (every 10 seconds)
time_windows = range(0, int(mp.times[-1]), 10)

# Loop through the signal_list and extract information for each time window
for window_start in time_windows:
    window_end = window_start + 10
    
    # Create a sub-DataFrame for the current time window
    sub_df = pd.DataFrame(index=range(10))  # Assuming 10 samples in each time window
    
    for signal in mp.signal_list:
        # Get the indices for the current time window
        mask = (signal.times >= window_start) & (signal.times < window_end)
        
        # Get the values within the current time window and interpolate for consistent time resolution
        signal_values = signal.values[mask]
        interpolated_values = pd.Series(signal_values).interpolate().values
        
        # Add the interpolated signal values to the sub-DataFrame
        sub_df[signal.name] = interpolated_values[:10]  # Truncate to 10 samples if needed
    
    # Calculate the mean and standard deviation for each signal in the current time window
    window_summary = []
    for signal_name in ["Unknown_0", "Unknown_1", "Unknown_2", "Unknown_3"]:
        mean_value = sub_df[signal_name].mean()
        std_value = sub_df[signal_name].std()
        # Round the standard deviation to two decimal places
        std_value = round(std_value, 2)
        window_summary.append([mean_value, std_value])
    
    # Append the window summary to the list
    summary_data.append({"Time Window": f"{window_start} - {window_end}", "Summary": window_summary})
    
    # Append the sub-DataFrame to the data dictionary with the window start as the key
    data_dict[f"{window_start} - {window_end}"] = sub_df

# Display the summary data
for window_data in summary_data:
    print(window_data["Time Window"])
    for i, signal_name in enumerate(["Unknown_0", "Unknown_1", "Unknown_2", "Unknown_3"]):
        mean_value, std_value = window_data["Summary"][i]
        print(f"{signal_name} [{mean_value}, {std_value}]")
    print()


# ### Exploring all IDS, signals and their signal values

# In[57]:


# List of IDs to explore
ids_to_explore = [6, 14, 37, 51, 58, 60, 61, 65, 117, 167, 186, 192, 204, 208, 215, 241, 244, 248, 253, 263, 293,
                  300, 304, 339, 354, 403, 412, 420, 426, 452, 458, 470, 485, 519, 526, 541, 560, 569, 622, 627,
                  628, 631, 640, 651, 661, 663, 675, 676, 683, 692, 695, 705, 722, 727, 737, 738, 778, 813, 837,
                  852, 870, 881, 930, 953, 961, 996, 1031, 1049, 1072, 1076, 1124, 1175, 1176, 1225, 1227, 1255,
                  1262, 1277, 1307, 1314, 1331, 1372, 1398, 1399, 1408, 1413, 1455, 1459, 1505, 1512, 1533, 1560,
                  1590, 1621, 1628, 1634, 1644, 1649, 1661, 1668, 1693, 1694, 1751, 1760, 1788]

# Loop through each ID and extract the signal values
for id_to_explore in ids_to_explore:
    mp = mapped_capture.mapped_payload_dict[id_to_explore]
    
    # Display the information about the ID and its signals
    print(f"Exploring ID: {id_to_explore}")
    print(f"Number of nonconstant bits: {len(mp.signal_list)}")
    for signal in mp.signal_list:
        print(f"Signal: {signal.name}, Start: {signal.start}, Length: {signal.length}")
        print(f"Signal values: {signal.values}")
    print()


# ### Exploring all IDs,time windows,  signals and their signal values

# In[58]:


import pandas as pd

# List of IDs to explore
ids_to_explore = [6, 14, 37, 51, 58, 60, 61, 65, 117, 167, 186, 192, 204, 208, 215, 241, 244, 248, 253, 263, 293,
                  300, 304, 339, 354, 403, 412, 420, 426, 452, 458, 470, 485, 519, 526, 541, 560, 569, 622, 627,
                  628, 631, 640, 651, 661, 663, 675, 676, 683, 692, 695, 705, 722, 727, 737, 738, 778, 813, 837,
                  852, 870, 881, 930, 953, 961, 996, 1031, 1049, 1072, 1076, 1124, 1175, 1176, 1225, 1227, 1255,
                  1262, 1277, 1307, 1314, 1331, 1372, 1398, 1399, 1408, 1413, 1455, 1459, 1505, 1512, 1533, 1560,
                  1590, 1621, 1628, 1634, 1644, 1649, 1661, 1668, 1693, 1694, 1751, 1760, 1788]

# Loop through each ID and extract the signal values
for id_to_explore in ids_to_explore:
    mp = mapped_capture.mapped_payload_dict[id_to_explore]
    
    # Display the information about the ID and its signals
    print(f"Exploring ID: {id_to_explore}")
    print(f"Number of nonconstant bits: {len(mp.signal_list)}")
    
    # Initialize a dictionary to store the extracted data for each time window
    data_dict = {}
    
    # Define the time windows (every 10 seconds)
    time_windows = range(0, int(mp.times[-1]), 10)
    
    # Loop through the signal_list and extract information for each time window
    for window_start in time_windows:
        window_end = window_start + 10
        
        # Create a sub-DataFrame for the current time window
        sub_df = pd.DataFrame(index=range(10))  # Assuming 10 samples in each time window
        
        for signal in mp.signal_list:
            # Get the indices for the current time window
            mask = (signal.times >= window_start) & (signal.times < window_end)
            
            # Get the values within the current time window and interpolate for consistent time resolution
            signal_values = signal.values[mask]
            interpolated_values = pd.Series(signal_values).interpolate(method='linear', limit_area='inside').values
            
            # Ensure that interpolated values match the length of the DataFrame
            interpolated_values = [interpolated_values[i] if i < len(interpolated_values) else None for i in range(10)]
            interpolated_values = pd.Series(interpolated_values).interpolate(method='linear', limit_area='inside').values
            
            # Add the interpolated signal values to the sub-DataFrame
            sub_df[signal.name] = interpolated_values
        
        # Append the sub-DataFrame to the data_dict with the window start as the key
        data_dict[(window_start, window_end)] = sub_df
    
    # Create a pandas DataFrame with the extracted data for the current ID
    df = pd.concat(data_dict, axis=1)
    
    # Convert columns to a regular Index with tuples representing the time windows and signal names
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    
    # Display the DataFrame for the current ID
    print(df)
    print()


# In[ ]:





# ### Mean and std for each signal per time window [mean,sd] -----------for ALL IDs

# In[59]:


import pandas as pd
import numpy as np

# List of IDs to explore
ids_to_explore = [6, 14, 37, 51, 58, 60, 61, 65, 117, 167, 186, 192, 204, 208, 215, 241, 244, 248, 253, 263, 293,
                  300, 304, 339, 354, 403, 412, 420, 426, 452, 458, 470, 485, 519, 526, 541, 560, 569, 622, 627,
                  628, 631, 640, 651, 661, 663, 675, 676, 683, 692, 695, 705, 722, 727, 737, 738, 778, 813, 837,
                  852, 870, 881, 930, 953, 961, 996, 1031, 1049, 1072, 1076, 1124, 1175, 1176, 1225, 1227, 1255,
                  1262, 1277, 1307, 1314, 1331, 1372, 1398, 1399, 1408, 1413, 1455, 1459, 1505, 1512, 1533, 1560,
                  1590, 1621, 1628, 1634, 1644, 1649, 1661, 1668, 1693, 1694, 1751, 1760, 1788]

# Function to extract mean and std for each column in a DataFrame
def extract_mean_std(df):
    return [df.mean().values, df.std().values]

# Loop through each ID and extract the signal values
for id_to_explore in ids_to_explore:
    mp = mapped_capture.mapped_payload_dict[id_to_explore]

    # Display the information about the ID and its signals
    print(f"ID {id_to_explore}")

    # Define the time windows (every 10 seconds)
    time_windows = range(0, int(mp.times[-1]), 10)

    # Loop through the signal_list and extract information for each time window
    for window_start in time_windows:
        window_end = window_start + 10
        print(f"{window_start} - {window_end}")

        # Create a sub-DataFrame for the current time window
        sub_df = pd.DataFrame(index=range(10))  # Assuming 10 samples in each time window

        for signal in mp.signal_list:
            # Get the indices for the current time window
            mask = (signal.times >= window_start) & (signal.times < window_end)

            # Get the values within the current time window and interpolate for consistent time resolution
            signal_values = signal.values[mask]
            interpolated_values = pd.Series(signal_values).interpolate(method='linear', limit_area='inside').values

            # Ensure that interpolated values match the length of the DataFrame
            interpolated_values = [interpolated_values[i] if i < len(interpolated_values) else None for i in range(10)]
            interpolated_values = pd.Series(interpolated_values).interpolate(method='linear', limit_area='inside').values

            # Add the interpolated signal values to the sub-DataFrame
            sub_df[signal.name] = interpolated_values

        # Compute the mean and std for the sub-DataFrame
        mean_std_array = extract_mean_std(sub_df)

        # Display the mean and std for each signal in the current time window
        for signal_name, (mean_val, std_val) in zip(sub_df.columns, zip(*mean_std_array)):
            print(f"{signal_name} [{mean_val:.2f}, {std_val:.2f}]")

        print()


# In[ ]:





# In[ ]:


annotating the attributes of the nodes


# In[4]:


import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# ... (Assuming you have defined and loaded the DataFrame 'df' as mentioned above)

# Define the time slice duration in seconds
time_slice_duration = 10.0

# Determine the number of time slices
num_slices = int(df['time'].max() / time_slice_duration) + 1

# Function to extract mean and std for each column in a DataFrame
def extract_mean_std(df):
    return [df.mean().values, df.std().values]

# Iterate over the time slices
for i in range(num_slices):
    start_time = i * time_slice_duration
    end_time = (i + 1) * time_slice_duration
    time_slice_label = f"Time Slice {i + 1}: ({start_time}, {end_time}]"
    time_slice_df = df[(df['time'] > start_time) & (df['time'] <= end_time)]

    # Build the graph for the current time slice
    G = nx.DiGraph()
    node_ids = time_slice_df['pid'].unique().tolist()
    G.add_nodes_from(node_ids)

    # Create a sub-DataFrame for the current time slice containing only the signal values
    sub_df = time_slice_df.drop(['time', 'pid'], axis=1)

    # Compute the mean and std for the sub-DataFrame
    mean_std_array = extract_mean_std(sub_df)

    # Annotate the nodes with the mean and std attributes
    for node_id, (mean_vals, std_vals) in zip(node_ids, zip(*mean_std_array)):
        G.nodes[node_id]['mean'] = mean_vals.tolist()
        G.nodes[node_id]['std'] = std_vals.tolist()

    for j in range(len(time_slice_df) - 1):
        source = time_slice_df.iloc[j]['pid']
        target = time_slice_df.iloc[j + 1]['pid']

        if G.has_edge(source, target):
            G[source][target]['weight'] += 1
        else:
            G.add_edge(source, target, weight=1)

    # Plot the graph
    plt.figure(i)
    pos = nx.spring_layout(G, seed=42)  # Adjust the layout algorithm as needed
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, edge_color='gray')

    # Add node attributes as labels to the graph
    labels = {}
    for node_id, attrs in G.nodes(data=True):
        labels[node_id] = f"{node_id}\nMean: {attrs['mean']}\nStd: {attrs['std']}"
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color='black')

    plt.title(time_slice_label)

plt.show()


# In[ ]:





# In[ ]:




