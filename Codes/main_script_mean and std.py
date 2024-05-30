import os
from CAN_objects.capture import MappedCapture

# Define the common base directory for the data files
base_directory = "C:\\Users\\willi\\Dropbox\\data-cancaptures"

# Define the common path for the capture.pkl file
cancap_filepath = os.path.join(base_directory, "capture.pkl")

# Specify the dataset-specific directories and .dbc file names
dataset_directories = [
    "road_ambient_dyno_drive_basic_long_050305_002000",
    # Add other dataset directories here if needed
]

dbc_files = [
    "anonymized_020822_030640.dbc",
    # Add other .dbc file names here if needed
]

def main():
    for dataset_dir, dbc_file in zip(dataset_directories, dbc_files):
        # Construct the full path for the .dbc file
        ground_truth_dbc_fpath = os.path.join(base_directory, "DBC", dbc_file)

        # Load the CAN capture data from the pickle file
        cancap = MappedCapture.unpickle(cancap_filepath)

        # Create a MappedCapture object from the CAN capture data and DBC file
        mapped_capture = MappedCapture.init_from_dbc(cancap, ground_truth_dbc_fpath)

        # Accessing the dictionary containing the payload mapping of the captured signals in 'mapped_capture'.
        payload_dict = mapped_capture.mapped_payload_dict

        # Retrieving the IDs from the mapped payload dictionary in 'mapped_capture'.
        ids_from_payload_dict = payload_dict.keys()

        
        
if __name__ == "__main__":
    main()
