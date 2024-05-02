import os
import mne
import numpy as np
from pybv import write_brainvision
import shutil

def copy_additional_files(vhdr_path, new_dir, base_filename):
    """Copy .vmrk and .eeg files associated with the .vhdr file to the new directory."""
    for ext in ['.vmrk', '.eeg']:
        original_file = vhdr_path.replace('.vhdr', ext)
        target_file = os.path.join(new_dir, base_filename + ext)
        if not os.path.exists(target_file):  # Check if the target file already exists
            shutil.copyfile(original_file, target_file)
            print(f"Copied {original_file} to {target_file}")
        else:
            print(f"Target file {target_file} already exists. Skipping.")

def convert_and_re_reference_vhdr(root_dir, move_dir, desired_channel_pairs):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.vhdr'):
                vhdr_path = os.path.join(dirpath, filename)
                relative_path = os.path.relpath(dirpath, root_dir)
                new_dir = os.path.join(move_dir, relative_path)
                base_filename = filename.replace('.vhdr', '_ref')
                new_vhdr_path = os.path.join(new_dir, base_filename + '.vhdr')
                
                if os.path.exists(new_vhdr_path):  # Skip if the target .vhdr file already exists
                    print(f"Target file {new_vhdr_path} already exists. Skipping.")
                    continue
                
                # Process the .vhdr file and save a new one
                raw = mne.io.read_raw_brainvision(vhdr_path, preload=True)
                new_data, new_channel_names = [], []
                for pair in desired_channel_pairs:
                    ch1, ch2 = pair  # Directly unpack the tuple
                    if ch1 in raw.ch_names and ch2 in raw.ch_names:
                        ch1_idx, ch2_idx = [raw.ch_names.index(ch) for ch in pair]
                        new_channel_data = raw._data[ch1_idx] - raw._data[ch2_idx]
                        new_data.append(new_channel_data)
                        new_channel_names.append('-'.join(pair))

                if new_data:
                    new_info = mne.create_info(new_channel_names, raw.info['sfreq'], ch_types='eeg')
                    new_raw = mne.io.RawArray(np.array(new_data), new_info)
                    new_raw.resample(256)
                    os.makedirs(new_dir, exist_ok=True)
                    write_brainvision(data=new_raw.get_data(), sfreq=new_raw.info['sfreq'], ch_names=new_raw.ch_names, 
                                      fname_base=base_filename, folder_out=new_dir, events=None, resolution=1e-9)
                    print(f"Processed and saved: {new_vhdr_path}")
                    copy_additional_files(vhdr_path, new_dir, base_filename)
                else:
                    print(f"No matching channels found for re-referencing in {filename}")

desired_channels_pairs = [
    ('Fp1', 'F7'), ('F7', 'T7'), ('T7', 'P7'), ('P7', 'O1'), 
    ('Fp1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1'), 
    ('Fz', 'Cz'), ('Cz', 'Pz'), 
    ('Fp2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2'),
    ('Fp2', 'F8'), ('F8', 'T8'), ('T8', 'P8'), ('P8', 'O2')
]

root_dir = '/Users/jeonsang-eon/sleep_data/'
move_dir = '/Users/jeonsang-eon/sleep_data_processed/'

convert_and_re_reference_vhdr(root_dir, move_dir, desired_channels_pairs)
