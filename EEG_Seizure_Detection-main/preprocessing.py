import mne 
import os
from data_management import get_folder_info
import matplotlib.pyplot as plt
import numpy as np

common_channels = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FZ-CZ', 'CZ-PZ', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8-0', 'P8-O2']
files_list = get_folder_info()

# Find start and end times of seizures in an edf file for a patient 
# Ex file_name = 'chb01_01.edf' 
def find_seizure_info(file_name):
    num_seizures = 0
    seizure_start_times = []
    seizure_end_times = []

    patient_folder = file_name.split('_')[0]

    #Corresponding text file for specified patient
    txt_file_name = f'{patient_folder}-summary.txt'

    txt_file_path = os.path.join(os.getcwd(), patient_folder, txt_file_name)

    found_target_file = False
    count = 0

    with open(txt_file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith('File Name: ' + file_name):
                found_target_file = True
            elif line.startswith('File Name:') and found_target_file:
                break
            elif found_target_file:
                if line.startswith('Number of Seizures in File:'):
                    num_seizures = int(line.split(':')[-1].strip())
                    count = 1 if num_seizures > 0 else 0
                elif line.startswith('Seizure Start Time:'):
                    seizure_start_times.append(int(line.split(':')[-1].strip().split()[0]))
                elif line.startswith('Seizure End Time:'):
                    seizure_end_times.append(int(line.split(':')[-1].strip().split()[0]))
                elif line.startswith('Seizure {} Start Time:'.format(count)):
                    seizure_start_times.append(int(line.split(':')[-1].strip().split()[0]))
                elif line.startswith('Seizure {} End Time:'.format(count)):
                    seizure_end_times.append(int(line.split(':')[-1].strip().split()[0]))
                    count += 1
    seizure_info = list(zip(seizure_start_times, seizure_end_times))
    if found_target_file == True:
        return seizure_info
    else:
        print('File does not exist')
        return None

#print(find_seizure_info('chb16_17.edf'))

# Patient num: 1,2,...,15
# File name: 'chb01_01.edf'
# Preictal duration (before seizure onset) defined in MINUTES

def preprocess(patient_num, file_name, preictal_duration):
    patient_folder = f'chb{patient_num:02d}'
    #file_name = 'chb{:02d}_{:02d}.edf'.format(patient_num, file_num)
    patient_file_path = files_list[patient_folder][file_name]

    # Filtering (Bandpass & Notch)
    raw = mne.io.read_raw_edf(patient_file_path, preload=True)
    raw.filter(0.1,127)
    raw.notch_filter(freqs=60)
    fs = raw.info['sfreq']

    # Remove channels not included in common channel list
    raw.pick_channels(common_channels)

    # Labeling 
    seizure_start_end_times = find_seizure_info(file_name)
    num_seizures = len(seizure_start_end_times)
    data_length = len(raw.get_data()[0]) # Number of samples for each channel (constant for all channels)
    labels = np.empty(data_length, dtype=object)

    for i in range(num_seizures):
        seizure_start, seizure_end = seizure_start_end_times[i]
        seizure_start_idx = int(fs*seizure_start)
        seizure_end_idx = int(fs*seizure_end)
        #length = seizure_end_idx - seizure_start_idx
        labels[seizure_start_idx:seizure_end_idx] = 'ict'

        if seizure_start_idx > preictal_duration*60*fs:
            preictal_start_idx = int(seizure_start_idx - fs*preictal_duration*60)
            labels[preictal_start_idx : seizure_start_idx] = 'pre'
        else:
            labels[:seizure_start_idx] = 'pre'

    return raw, labels

#preprocess(16, 'chb16_16.edf')
#band_signals = preprocess(1, 1)
# num_samples = 1000
# for band in bands.keys():
#     plt.plot(band_signals[band][0][:num_samples])
# plt.show()
 
