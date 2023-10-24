# embedding_generation.py
from imports import *
from utils import extract_mean_std_for_id 

def generate_node_embeddings(all_graphs):
    all_node_embeddings = []
    for idx, G in enumerate(tqdm(all_graphs, desc="Generating Node Embeddings")):
        node2vec = Node2Vec(G, dimensions=64, walk_length=15, num_walks=100, workers=19, p=1.5, q=0.5)
        model = node2vec.fit(window=10, min_count=1, batch_words=7)
        
        node_embeddings = {}
        for node in G.nodes():
            if node in model.wv:
                node_emb = model.wv[node]
                node_embeddings[node] = node_emb
            else:
                node_embeddings[node] = np.zeros(model.vector_size)
        
        print(f"Processed graph {idx+1}: Nodes - {G.number_of_nodes()}, Edges - {G.number_of_edges()}")

        all_node_embeddings.append(node_embeddings)
    
    return all_node_embeddings

def combine_embeddings_with_attributes(all_graphs, all_node_embeddings, time_slice_duration, mapped_capture):
    all_combined_node_embeddings = []

    # Calculate the maximum number of attributes across all nodes and all graphs
    max_attributes = max(
        [len(extract_mean_std_for_id(node, idx * time_slice_duration, (idx + 1) * time_slice_duration, mapped_capture))
         for idx, graph in enumerate(all_graphs)
         for node in graph.nodes()]
    )

    # Iterate over all graphs and their index
    for idx, G in tqdm(enumerate(all_graphs), desc="Combining Node Embeddings with Attributes"):
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

    return all_combined_node_embeddings


def compute_average_embeddings(all_combined_node_embeddings, highest_signal_count):
    """
    Compute the average embeddings for each graph based on the combined node embeddings.

    Parameters:
    - all_combined_node_embeddings: List of dictionaries containing combined node embeddings for each graph.
    - highest_signal_count: Integer indicating the highest number of signals across all nodes.

    Returns:
    - concatenated_embeddings: Dictionary containing average embeddings for each graph.
    """
    # Calculate the new dimension based on 64 + (2 attributes * highest number of signals)
    new_dimension = 64 + (2 * highest_signal_count)

    # Initialize a dictionary to store concatenated embeddings
    concatenated_embeddings = {}

    # Iterate through all graphs in the capture
    for graph_embeddings in all_combined_node_embeddings:
        # Initialize variables to accumulate embeddings and count nodes
        total_embeddings = np.zeros((new_dimension,))
        num_nodes = 0

        # Iterate through node embeddings in the current graph
        for node_id, node_embedding in graph_embeddings.items():
            # Pad node embeddings to the new dimension
            padded_embeddings = np.pad(node_embedding, (0, new_dimension - len(node_embedding)), mode='constant')
            
            # Accumulate the padded embeddings
            total_embeddings += padded_embeddings
            num_nodes += 1

        # Calculate average embeddings for the current graph
        if num_nodes > 0:
            average_embeddings = total_embeddings / num_nodes
        else:
            # Handle the case where there are no nodes in the graph
            average_embeddings = np.zeros((new_dimension,))
        
        # Store the average embeddings in the dictionary
        concatenated_embeddings[len(concatenated_embeddings)] = average_embeddings

    return concatenated_embeddings

def create_dataframe_from_embeddings(concatenated_embeddings):
    """
    Create a DataFrame from the concatenated embeddings.
    
    Parameters:
    - concatenated_embeddings: Dictionary containing the embeddings for each graph.
    
    Returns:
    - df_benign: DataFrame with embeddings and labels.
    """
    # Create a list to store the embeddings and labels
    data = []

    # Iterate through the benign embeddings
    for graph_id, embedding in concatenated_embeddings.items():
        # Append the embedding and label to the data list
        data.append((embedding, '0'))

    # Create a DataFrame
    df_benign = pd.DataFrame(data, columns=['Embedding', 'Label'])

    return df_benign

def display_combined_embeddings(embeddings, num_to_display=5):
    """
    Display the combined embeddings for the specified number of nodes.

    Parameters:
    - embeddings: The combined node embeddings.
    - num_to_display: The number of nodes for which to display the embeddings.
    """
    for node, embedding in list(embeddings.items())[:num_to_display]:
        print(f"Node: {node}\nEmbedding: {embedding}\n")
