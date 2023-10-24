# data_processing.py
from imports import *

def make_can_df(log_filepath):
    can_df = pd.read_fwf(
        log_filepath, delimiter = ' '+ '#' + '('+')',
        skiprows = 1, skipfooter=1,
        usecols = [0, 2, 3],
        dtype = {0: 'float64', 1: str, 2: str},
        names = ['time', 'pid', 'data'])

    can_df.pid = can_df.pid.apply(lambda x: int(x, 16))
    can_df.data = can_df.data.apply(lambda x: x.zfill(16))
    can_df.time = can_df.time - can_df.time.min()
    return can_df[can_df.pid <= 0x700]

def generate_graphs_from_data(df, window_size, offset):
    all_graphs = []
    num_slices = int((df['time'].max() - df['time'].min()) / offset)
    for i in range(num_slices):
        start_time = df['time'].min() + i * offset
        end_time = start_time + window_size
        time_slice_df = df[(df['time'] >= start_time) & (df['time'] < end_time)]
        
        G = nx.DiGraph()
        node_ids = time_slice_df['pid'].unique().tolist()
        G.add_nodes_from(node_ids)
        for j in range(len(time_slice_df) - 1):
            source = time_slice_df.iloc[j]['pid']
            target = time_slice_df.iloc[j + 1]['pid']
            if G.has_edge(source, target):
                G[source][target]['weight'] += 1
            else:
                G.add_edge(source, target, weight=1)

        if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
            print(f"Graph for slice {i+1} is empty or has no edges!")

        all_graphs.append(G)

    return all_graphs

def process_dataframe(df, window_size, offset):
    df_sorted = df.sort_values('time')
    df_sorted['time'] = df_sorted['time'].round(2)

    num_slices = ceil((df_sorted['time'].max() - df_sorted['time'].min()) / offset)
    for i in range(num_slices):
        start_time = df_sorted['time'].min() + i * offset
        end_time = start_time + window_size
        time_slice_label = f"Time Slice {i + 1}: ({start_time}, {end_time}]"
        time_slice_df = df_sorted[(df_sorted['time'] >= start_time) & (df_sorted['time'] < end_time)]
        print(f"{time_slice_label}\n{time_slice_df}\n")