# Detecting Masquerade Attacks in Controller Area Networks Using Graph Machine Learning

## Project Overview

### Main Objective
Design and develop a unified framework to detect both masquerade attacks in the CAN bus that can be deployed on edge devices. We aim to model CAN message streams as CAN graph messages, embedding both node and edge attributes. From this, we can train a model using normal data to detect attacks in the test dataset.

### Key Questions & Hypothesis
- **Question:** Can we detect masquerade attacks in the CAN bus using a graph ML?
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
   - Train the Graph ML framework for IDS in CAN graph messages.
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
   - Combining embeddings with signal attributes after timeseries signal extraction.
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

- **Process All Reverse Light Off Attack Files (`process_all_reverse_light_off_attack_files.py`)**:
   - Script to process all reverse light off masquerade attack files.
   - Reads, preprocesses, and extracts features from the CAN logs.
   - Generates node embeddings and combines them with signal attributes.
   - Computes average embeddings and converts them into DataFrames.
   - Saves the results to disk as CSV and PKL files.

- **Process All Reverse Light On Attack Files (`process_all_reverse_light_on_attack_files.py`)**:
   - Script to process all reverse light on masquerade attack files.
   - Similar structure and functionality to the reverse light off attack processing script.

- **Process Max Engine Coolant Temp Attack Masquerade File (`process_max_engine_coolant_temp_attack_masquerade_file.py`)**:
   - Script to process max engine coolant temperature attack masquerade files.
   - Reads and preprocesses the CAN logs.
   - Extracts features and generates node embeddings.
   - Combines embeddings with signal attributes and computes average embeddings.
   - Converts embeddings to DataFrames and saves results as CSV and PKL files.

- **Process All Max Speedometer Attack Files (`process_all_max_speedometer_attack_files.py`)**:
   - Script to process all max speedometer masquerade attack files.
   - Reads, preprocesses, and extracts features from the CAN logs.
   - Generates node embeddings and combines them with signal attributes.
   - Computes average embeddings and converts them into DataFrames.
   - Saves the results to disk as CSV and PKL files.

- **Main Script (`main_script.py`)**:
   - The main script to run experiments.
   - Integrates various modules and functions to execute the complete workflow for detecting masquerade attacks in CAN data.

- **Run Experiments (`run_experiments.py`)**:
   - Script to execute the entire experiment pipeline.
   - Automates data preprocessing, feature extraction, embedding generation, and result analysis.
   - Facilitates running experiments with different configurations and datasets.

## Dataset
Our framework is evaluated using the Real ORNL Automotive Dynamometer (ROAD) dataset, developed by the Oak Ridge National Laboratory (ORNL) \cite{verma2022addressing}. The ROAD dataset includes CAN data from real vehicles with verified fabrication and simulated masquerade attacks, providing a realistic environment for testing CAN security methods. The dataset includes 3.5 hours of recorded data, with 3 hours for training and 30 minutes for testing, covering various driving scenarios. It includes five masquerade attacks: correlated signal, max engine, max speedometer, reverse light off, and reverse light on attacks. For more details, refer to the [ROAD dataset paper](https://doi.org/10.1371/journal.pone.0296879).



## Getting Started

### Running the Code(the window sizes and offsets can be altered depending on overlaps and time slices)
#### Benign Files:
1. **To run only the first benign file, which is `ambient_dyno_drive_basic_long.log`:**
   ```shell
   python ambient_dyno_drive_basic_long.py --window-size 10 --offset 10 --pkl-folder road_ambient_dyno_drive_basic_long_050305_002000 "C:\Users\willi\OneDrive\Desktop\Research\oak_ridge_in_vehicle\road\ambient\ambient_dyno_drive_basic_long.log"


1. **To run all benign files:**
   ```shell
   python process_all_files.py --window-size 4 --offset 4


  
1. **To process all correlated masquerade attack files:**
   ```shell
   python process_all_correlated_masquerade_attack_files.py --window-size 4 --offset 4


2. **To process all correlated masquerade attack files:

   ```shell

python process_all_correlated_masquerade_attack_files.py --window-size 4 --offset 4

3. **To process all reverse light off masquerade attack files:

   ```shell

     python process_all_reverse_light_off_attack_files.py --window-size 4 --offset 4

4.**To process all reverse light on masquerade attack files:

   ```shell

    python process_all_reverse_light_on_attack_files.py --window-size 4 --offset 4

5. **To process max engine coolant temperature attack masquerade files:

  ```shell

python process_max_engine_coolant_temp_attack_masquerade_file.py --window-size 4 --offset 4

6. **To process all max speedometer masquerade attack files:

```shell

python process_all_max_speedometer_attack_files.py --window-size 4 --offset 4
