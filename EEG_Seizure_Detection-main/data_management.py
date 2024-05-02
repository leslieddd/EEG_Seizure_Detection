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
