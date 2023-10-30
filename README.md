# CAN Bus Attack Detection Using GNN

## Project Overview

### Main Objective
Design and develop a unified framework to detect both fabrication/masquerade attacks in the CAN bus that can be deployed on edge devices. We aim to model CAN message streams as CAN graph messages, embedding both node and edge attributes. From this, we can train a model using normal data to detect attacks in the test dataset.

### Key Questions & Hypothesis
- **Question:** Can we detect fabrication/masquerade attacks in the CAN bus using a GNN graph classification approach?
- **Hypothesis:** 
  1. CAN messages graphs (embedding both node/edge attributes) may characterize normal/attack conditions in CAN.
  2. Fabrication/masquerade attacks scenarios can be framed as a GNN graph classification framework.

### Workflow
1. **Phase 1 - Literature Review**
   - Familiarization with CAN protocol, in-vehicle security, CAN IDS, and Graph ML.
   - Experimentation with the ROAD dataset.

2. **Phase 2 - Building & Annotating CAN Message Graphs**
   - Parse CAN data to build CAN message graphs.
   - Partition CAN streaming data into temporal CAN message graphs.
   - Embed attributes from CAN signals into nodes and frequency of messages into edges.

3. **Phase 3 - Experiments with GNN for Graph Classification**
   - Build training datasets based on CAN graph messages.
   - Train the GNN framework for IDS in CAN graph messages.
   - Test the trained GNN framework on attack captures and compute classification metrics.

## Code Structure

### Python Scripts:
- **Data Processing (`data_processing.py`)**:
   - Functions for reading and preprocessing the CAN logs.
   - Functions to partition the CAN data into temporal graphs.

- **Utils (`utils.py`)**:
   - Utility functions to load mapped captures, extract mean and standard deviation for specific IDs, and other helper functions.

- **Embedding Generation (`embedding_generation.py`)**:
   - Functions to generate node embeddings using Node2Vec.
   - Combining embeddings with signal attributes.
   - Computing average embeddings and converting them into DataFrames.

- **Imports (`imports.py`)**:
   - This script includes all necessary imports for the entire project. It consolidates the libraries and modules needed, like `os`, `argparse`, `pandas`, `networkx`, `node2vec`, and others, making it easier to manage dependencies.

- **Ambient Dyno Drive Basic Long (`ambient_dyno_drive_basic_long.py`)**:
   - This script is the foundational file that initiates the entire workflow for CAN data processing. 
   - It first parses the CAN logs and processes the data into a structured DataFrame.
   - It then generates graphs from the processed data, and these graphs are used to create node embeddings.
   - The node embeddings are combined with other attributes like signal data, and this combined data is then utilized to compute average embeddings.
   - The script also provides functionality to visualize the combined embeddings, find nodes with the highest number of signals, and save results to disk in the form of CSV and PKL files.

- **Process All Files (`process_all_files.py`)**:
   - Script to process multiple benign CAN log files.

- **Process All Correlated Masquerade Attack Files (`process_all_correlated_masquerade_attack_files.py`)**:
   - Script to process all the correlated masquerade attack files.

## Getting Started

### Running the Code(the window sizes and offsets can be altered depending on overlaps and time slices)
#### Benign Files:
1. To run only the first benign file which is `ambient_dyno_drive_basic_long.log`:
   ```shell
   python ambient_dyno_drive_basic_long.py --window-size 10 --offset 10 --pkl-folder road_ambient_dyno_drive_basic_long_050305_002000 "C:\Users\willi\OneDrive\Desktop\Research\oak_ridge_in_vehicle\road\ambient\ambient_dyno_drive_basic_long.log"
2. **To run all benign files:
   ```shell
python process_all_files.py --window-size 4 --offset 4
3. **Process All Correlated Masquerade Attack Files
   ```shell
python process_all_correlated_masquerade_attack_files.py --window-size 4 --offset 4
