import os
from preprocessing import preprocess

# Define paths
root_dir_path = '/Users/jeonsang-eon/sleep_data/'
output_root_dir_path = '/Users/jeonsang-eon/sleep_data_processed/'

sub_names = [f'sub-{i:02d}' for i in range(1, 34)]

def extract_session_name(filename):
    # This function assumes the filename format is 'sub-xx_[session]_eeg.vhdr'
    parts = filename.split('_')
    if len(parts) > 2:
        # Join all parts except the first and last to handle session identifiers with underscores
        return '_'.join(parts[1:-1])
    return None


for sub_name in sub_names:
    # Output directory path
    output_dir_path = os.path.join(output_root_dir_path, sub_name)

    # Ensure the output directory exists
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    # Data directory path
    data_dir_path = os.path.join(root_dir_path, sub_name)

    if os.path.exists(data_dir_path):
        for entry in os.listdir(data_dir_path):
            if entry.endswith('.vhdr'):
                data_file_path = os.path.join(data_dir_path, entry)
                session_name = extract_session_name(entry)
                
                # Update label file path to include the specific session name
                label_file_path = os.path.join(root_dir_path, 'sourcedata', f'{sub_name}-sleep-stage.tsv')
                
                # Call a processing function or a dummy function
                preprocess(sub_name, session_name, data_file_path, label_file_path, output_dir_path)
    else:
        print(f"Directory does not exist: {data_dir_path}")