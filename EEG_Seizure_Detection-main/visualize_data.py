import os
import mne
import matplotlib.pyplot as plt
from data_exploration_mne import get_folder_info

all_patient_folders = [folder for folder in os.listdir() if os.path.isdir(folder) and 'chb' in folder]

common_channels = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FZ-CZ', 'CZ-PZ', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8-0', 'P8-O2']

def visualize_patient_data(patient_num, num_samples, list_files, file_numbers, channel_names=common_channels, start_sample=0):
    file_names = ['chb{:02d}_{:02d}.edf'.format(patient_num, file_number) for file_number in file_numbers]
    patient_folder = f'chb{patient_num:02d}'
    if patient_folder in all_patient_folders:
        patient_files = list_files[patient_folder]
        for file_name in file_names:
            if file_name in patient_files:
                plt.figure()
                patient_file_path = patient_files[file_name]
                raw = mne.io.read_raw_edf(patient_file_path, preload=True)
                data = raw.get_data()
                i = 0
                for channel_num in range(len(data)):
                    channel_name = raw.ch_names[channel_num]
                    if channel_name in (common_channels and channel_names):
                        i+=1 
                        channel_data = data[channel_num]
                        plt.subplot(len(channel_names), 1, i)
                        end_sample = start_sample + num_samples
                        plt.plot(channel_data[start_sample:end_sample])
                        plt.ylabel(channel_name, rotation=0, labelpad=30)
                plt.show()
            else:
                print("Missing file '{}' in folder '{}'".format(file_name, patient_folder))

    else:
        print("No data folder 'chb{:02d}' for patient {} in local directory.".format(patient_num, patient_num))


# Function testing 
#files_list = get_folder_info()
#channels_to_plot = ['FP1-F7', 'F7-T7']
#visualize_patient_data(1, 1000, files_list, [3,4], channels_to_plot)