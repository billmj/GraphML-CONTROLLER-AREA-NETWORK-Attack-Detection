
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
   - Functions to generate edge embeddings.

### Dataset:
- **ROAD Dataset**: This dataset contains multiple hours of recorded data, with 3 hours for training and 30 minutes for testing, covering various driving scenarios. It includes five masquerade attacks: correlated signal, max engine, max speedometer, reverse light off, and reverse light on attacks. For more details, refer to the [ROAD dataset paper](https://doi.org/10.1371/journal.pone.0296879).

## Getting Started

### Running the Code (the window sizes and offsets can be altered depending on overlaps and time slices)

#### Benign Files:
1. **To run only the first benign file, which is `ambient_dyno_drive_basic_long.log`:**
   ```shell
   python ambient_dyno_drive_basic_long.py --window-size 10 --offset 10 --pkl-folder road_ambient_dyno_drive_basic_long_050305_002000 "C:\Users\willi\OneDrive\Desktop\Research\oak_ridge_in_vehicle\road\ambient\ambient_dyno_drive_basic_long.log"
   ```

2. **To run all benign files:**
   ```shell
   python process_all_files.py --window-size 4 --offset 4
   ```

#### Attack Files:
3. **To process all `correlated masquerade attack files`:**
   ```shell
   python process_all_correlated_masquerade_attack_files.py --window-size 4 --offset 4
   ```

4. **To process all `reverse light off masquerade attack files`:**
   ```shell
   python process_all_reverse_light_off_attack_files.py --window-size 4 --offset 4
   ```

5. **To process all `reverse light on masquerade attack files`:**
   ```shell
   python process_all_reverse_light_on_attack_files.py --window-size 4 --offset 4
   ```

6. **To process all `max engine coolant temperature masquerade attack files`:**
   ```shell
   python process_max_engine_coolant_temp_attack_masquerade_file.py --window-size 4 --offset 4
   ```

7. **To process all `max speedometer masquerade attack files`:**
   ```shell
   python process_all_max_speedometer_attack_files.py --window-size 4 --offset 4
   ```

