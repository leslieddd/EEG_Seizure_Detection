import os
import matplotlib.pyplot as plt
import mne

common_channels = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FZ-CZ', 'CZ-PZ', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8-0', 'P8-O2']
#all_patient_folders = [folder for folder in os.listdir() if os.path.isdir(folder) and 'chb' in folder]

patient_numbers = [1,2,3,6,16]

data_directory = os.getcwd()    # stored all data ('chb01', 'chb02', ...) in the same directory as this .py file 
#patients_with_data = []
patient_folders = []


files_list = {}

for patient_number in patient_numbers:
    patient_folder = f'chb{patient_number:02d}'
    files_list[patient_folder] = []
    folder_path = os.path.join(data_directory, patient_folder)
    if os.path.isdir(folder_path) and os.path.exists(folder_path):
        files = os.listdir(folder_path)
        for file_name in files:
            if file_name.endswith('.edf'):
                file_path = os.path.join(folder_path, file_name)
                files_list[patient_folder].append((file_name, file_path))
        #patients_with_data.append(patient_number)
        patient_folders.append(patient_folder)
    else:
        print(f'Error: Patient folder {patient_folder} not found! Patient {patient_number} data may be missing.')


# VISUALIZE DATA FOR A PATIENT

patient_number = 1
num_samples = 1000

patient_folder = f'chb{patient_number:02d}'
print(patient_folder)
if patient_folder in patient_folders:
    patient_file_paths = [tup[1] for tup in files_list[patient_folder]]
    print(patient_file_paths)
    for num in range(len(patient_file_paths)):
        plt.figure()
        patient_file_path = patient_file_paths[num]
        print(patient_file_path)
        raw = mne.io.read_raw_edf(patient_file_path, preload=True)
        data = raw.get_data()
        i = 0
        for channel_num in range(len(data)):
            channel_name = raw.ch_names[channel_num]
            print(channel_name)
            if channel_name in common_channels:
                i+=1 
                plt.subplot(len(common_channels), 1, i)
                plt.plot(data[channel_num][:num_samples])
                plt.ylabel(channel_name, rotation=0, labelpad=30)
        plt.show()

