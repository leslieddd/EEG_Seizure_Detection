import os
import matplotlib.pyplot as plt
import mne

all_patient_folders = [folder for folder in os.listdir() if os.path.isdir(folder) and 'chb' in folder]
data_directory = os.getcwd() # stored all data ('chb01', 'chb02', ...) in the same directory as this .py file 
common_channels = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FZ-CZ', 'CZ-PZ', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8-0', 'P8-O2']

def get_folder_info():
    list_files = {}
    for patient_folder in all_patient_folders:
        list_files[patient_folder] = {}
        folder_path = os.path.join(data_directory, patient_folder)
        if os.path.isdir(folder_path) and os.path.exists(folder_path):
            files = os.listdir(folder_path)
            for file_name in files:
                if file_name.endswith('.edf'):
                    file_path = os.path.join(folder_path, file_name)
                    list_files[patient_folder][file_name] = file_path
    return list_files

def visualize_patient_data(patient_num, num_samples, list_files, file_numbers, channel_names=common_channels):
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
                        plt.plot(channel_data[:num_samples])
                        plt.ylabel(channel_name, rotation=0, labelpad=30)
                plt.show()
            else:
                print("Missing file '{}' in folder '{}'".format(file_name, patient_folder))

    else:
        print("No data folder 'chb{:02d}' for patient {} in local directory.".format(patient_num, patient_num))



files_list = get_folder_info()
# channels_to_analyze = ['FP1-F7', 'F7-T7']
# visualize_patient_data(1, 1000, files_list, [1,2], channels_to_analyze)


bands = {'Delta':(0.5,4), 'Theta':(4,8), 'Alpha':(8,12), 'Beta':(12,30), 'Gamma':(30,120)}

def preprocess(patient_num, file_num):
    patient_folder = f'chb{patient_num:02d}'
    file_name = 'chb{:02d}_{:02d}.edf'.format(patient_num, file_num)
    patient_file_path = files_list[patient_folder][file_name]

    # Filtering 
    raw = mne.io.read_raw_edf(patient_file_path, preload=True)
    raw.filter(0.1,127)
    raw.notch_filter(freqs=60)

    # Band separation
    band_signals = {}
    for band_name, (low_cutoff, high_cutoff) in bands.items():
        band_raw = raw.copy().filter(low_cutoff,high_cutoff)
        band_signals[band_name] = band_raw.get_data()

    # Ictal period labeling
    #find_seizure_info(file_name)

    return band_signals

band_signals = preprocess(1, 1)
num_samples = 1000
for band in bands.keys():
    plt.plot(band_signals[band][0][:num_samples])
plt.show()

def find_seizure_info(file_name):
    seizure_count = 0
    seizure_start_times = []
    seizure_end_times = []

    patient_folder = file_name.split('_')[0]
    txt_file_name = f'{patient_folder}-summary.txt'

    txt_file_path = os.path.join(os.getcwd(), patient_folder, txt_file_name)

    found_target_file = False

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
                elif line.startswith('Seizure Start Time:'):
                    seizure_start_times.append(int(line.split(':')[-1].strip().split()[0]))
                elif line.startswith('Seizure End Time:'):
                    seizure_end_times.append(int(line.split(':')[-1].strip().split()[0]))
    seizure_info = list(zip(seizure_start_times, seizure_end_times))
    return num_seizures, seizure_info

num_seizures, seizure_info = find_seizure_info('chb01_03.edf')
print(num_seizures)
print(seizure_info)
