import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import mne
import json



F_SAM=64
SEG_DUR=30
ORDEDRED_CHANNEL_PAIRS = [
    ('Fp1', 'F7'), ('F7', 'T7'), ('T7', 'P7'), ('P7', 'O1'), 
    ('Fp1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1'), 
    ('Fz', 'Cz'), ('Cz', 'Pz'), 
    ('Fp2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2'),
    ('Fp2', 'F8'), ('F8', 'T8'), ('T8', 'P8'), ('P8', 'O2')
]
def re_reference_eeg(raw):
    new_data = []
    new_channel_names = []
    for ch1, ch2 in ORDEDRED_CHANNEL_PAIRS:
        if ch1 in raw.ch_names and ch2 in raw.ch_names:
            ch1_data, ch2_data = raw[ch1][0], raw[ch2][0]
            ref_data = ch1_data - ch2_data
            new_data.append(ref_data)
            new_channel_names.append(f'{ch1}-{ch2}')
    return np.array(new_data), new_channel_names


def preprocess(sub_name, session_name, data_file_path, label_file_path, output_dir_path):
    try:
        raw = mne.io.read_raw_brainvision(data_file_path, preload=True, verbose='ERROR')
        raw.resample(F_SAM)
    except Exception as e:
        print(f"Failed to load or resample the EEG data: {e}")
        return

    new_data, new_channel_names = re_reference_eeg(raw)
    samples_per_segment = F_SAM * SEG_DUR

    for i in range(new_data.shape[2] // samples_per_segment):
        start_sample = i * samples_per_segment
        end_sample = start_sample + samples_per_segment
        segment_data = new_data[:, :, start_sample:end_sample].squeeze()
        start_time = start_sample // F_SAM

        labels = pd.read_csv(
                        label_file_path, 
                        delimiter='\t', 
                        usecols=['session', 'epoch_start_time_sec', '30-sec_epoch_sleep_stage']
                        )
        # Find the label for the segment based on start time
        # Correct the condition to filter the DataFrame for the label
        label_row = labels[(labels['epoch_start_time_sec'] == start_time) & (labels['session'] == session_name)]

        # Check if the label_row is empty and assign the label accordingly
        label = label_row['30-sec_epoch_sleep_stage'].iloc[0] if not label_row.empty else 'Unknown'

        if(label=='Unknown'):
            continue

        specPack = []
        for channel_data in segment_data:
            f, t, Sxx = spectrogram(channel_data, fs=F_SAM, nperseg=256, noverlap=256-15)
            specPack.append(10*np.log10(Sxx))

        if len(specPack) > 0:
            specPack = np.array(specPack)
            

            
            # Save
            if not os.path.exists(output_dir_path):
                os.makedirs(output_dir_path)
            npy_file_name = f"{sub_name}_{session_name}_{start_time}s.npy"
            np.save(os.path.join(output_dir_path, npy_file_name), specPack)

            dimension=specPack.shape
            meta_data = {
                'channels': ORDEDRED_CHANNEL_PAIRS,
                'start_time': start_time,
                'duration': SEG_DUR,
                'parent_file': data_file_path,
                'label': label,
                'dimension': dimension
            }

            meta_file_name=f"{sub_name}_{session_name}_{start_time}s.json"
            with open(os.path.join(output_dir_path, meta_file_name), 'w') as f:
                json.dump(meta_data, f)
            print(f"Saved spectrogram and metadata for start time {start_time}s with label {label}.")

        else:
            print(f"No data processed for segment starting at {start_sample/F_SAM} seconds.")
