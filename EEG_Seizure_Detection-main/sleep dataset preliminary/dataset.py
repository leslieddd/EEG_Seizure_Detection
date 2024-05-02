import os
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader
import torch
import sympy as sp


# # Dictionary of label distributions
# LABEL_DSTR = {
#     'W': 3114, '1': 2105, '2': 1243, '3': 42,
#     '1 (unscorable)': 6, '2 (unscorable)': 10, '3 (unscorable)': 11, '2 or 3 (unscorable)': 3, 
#     'W (uncertain)': 38, '1 (uncertain)': 41, '2 (uncertain)': 8, '3 (uncertain)': 1,
#     'unscorable': 2, 'Unscorable': 1
# }

# # Labels considered for basic prior calculation
# CERTAIN_LABELS = ['W', '1', '2', '3']

# # Calculate the sum of counts for certain labels
# tempSum = sum(LABEL_DSTR[label] for label in CERTAIN_LABELS)
# PRIOR = {label: LABEL_DSTR[label] / tempSum for label in CERTAIN_LABELS}

# def uncertain_vec(label):
#     """ Calculates transformed probabilities for given label considering the effect of other labels. """
#     p = sp.symbols('p')
#     p_label = PRIOR[label]
#     other_labels = {k: v for k, v in PRIOR.items() if k != label}
#     max_other_p = max(other_labels.values())

#     # Solve the probability transformation equation
#     equation = sp.Eq(p, (1 - p) / (1 - p_label) * max_other_p)
#     solution = sp.solve(equation, p)
#     p_value = min([sol.evalf() for sol in solution if sol.is_real and sol >= 0], default=0)

#     # Apply the transformation to all probabilities
#     transformed_probabilities = {k: (v * (1 - p_value) / (1 - p_label) if k != label else p_value) for k, v in PRIOR.items()}
#     return [transformed_probabilities[k] for k in CERTAIN_LABELS]

# # Dictionary for storing the solution vectors for each label type
# TAR_DICT = {
#     'W': [1, 0, 0, 0],
#     '1': [0, 1, 0, 0],
#     '2': [0, 0, 1, 0],
#     '3': [0, 0, 0, 1],
#     'unscorable': list(PRIOR.values()), 
#     'Unscorable': list(PRIOR.values())
# }

# # Update dictionary with vectors from uncertain vector calculations
# for label in CERTAIN_LABELS:
#     uncertain_vec_result = uncertain_vec(label)
#     for uncertain_label in [f'{label} (unscorable)', f'{label} (uncertain)']:
#         TAR_DICT[uncertain_label] = uncertain_vec_result

# # Handle the special case '2 or 3 (unscorable)'
# TAR_DICT['2 or 3 (unscorable)'] = [0, 0, 0.5, 0.5]

# #########################################################################################
# #########################################################################################
# #########################################################################################
# #########################################################################################
# #########################################################################################
# #########################################################################################

# # SOL_DICT Initiated
CERTAIN_LABELS = ['W', '1', '2', '3']
TAR_DICT = {
    'W': [1, 0, 0, 0],
    '1': [0, 1, 0, 0],
    '2': [0, 0, 1, 0],
    '3': [0, 0, 0, 1]
}

def num_of_proper_labeled_file(ses_path):
    tot=0
    files = os.listdir(ses_path)
    for f in files:
        if f.endswith('.json'):
            full_path = os.path.join(ses_path, f)
            with open(full_path, 'r') as file:
                label_data = json.load(file)
                label_str = label_data['label']
                if label_str in CERTAIN_LABELS:
                    tot+=1
    return tot
                
class EEGSesDataset(Dataset):
    def __init__(self, session_path):
        self.times = []
        self.session_path = session_path
        self.data=[]
        self.labels=[]

        files = os.listdir(session_path)
        for f in files:
            if f.endswith('.json'):
                full_path = os.path.join(session_path, f)
                with open(full_path, 'r') as file:
                    label_data = json.load(file)
                    label_str = label_data['label']
                    if label_str not in CERTAIN_LABELS:
                        continue
                    time_stamp = int(f.split('.')[0])
                    self.times.append(time_stamp)
                    self.labels.append(TAR_DICT[label_str])
        
        max_time = max(self.times)

        for i in range(0, max_time + 1, 30):
            data_path = os.path.join(self.session_path, f"{i}.npy")
            if os.path.exists(data_path):
                data = np.load(data_path)
                self.data.append(data)
            else:
                print(f"data path{data_path} is wrong")
            
        

    def __len__(self):
        return len(self.times)

    def __getitem__(self, idx):
        current_time = self.times[idx]
        data_idx = current_time // 30  # Index of the data in self.data
        # print(f"current_time: {current_time}")
        
        if(data_idx>len(self)):
            raise IndexError("Index out of range")
        
        # print(f"data_idx: {data_idx}")

        # Assuming self.data is a list of numpy arrays
        data = np.expand_dims(self.data[0], axis=0) if data_idx == 0 \
            else self.data[0:data_idx]

        label = self.labels[idx]

        data = torch.tensor(data, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long) if label is not None else torch.tensor(0, dtype=torch.long)

        return data, label



# # Ensure that CERTAIN_LABELS and TAR_DICT are properly defined before using this class.
# ses_path='/Users/jeonsang-eon/sleep_data_processed/sub-11/task-rest_run-2'
# ex=EEGSesDataset(ses_path)
# print(len(ex))
# print(ex[10][0].shape)



class EEGSubjDataset(Dataset):
    def __init__(self, subj_dir):
        self.session_paths = []  # This will store session paths
        self.index_ranges = []
        self.current_session_dataset = None
        self.current_session_start_idx = -1  # Initialized to -1 to indicate no session is loaded
        total_index = 0

        for session_name in sorted(os.listdir(subj_dir)):
            session_path = os.path.join(subj_dir, session_name)
            if os.path.isdir(session_path):
                # Estimate the number of valid files without loading the entire dataset
                num_files = num_of_proper_labeled_file(session_path)
                if num_files > 0:
                    self.session_paths.append(session_path)
                    self.index_ranges.append((total_index, total_index + num_files))
                    total_index += num_files

    def __len__(self):
        return sum(end - start for start, end in self.index_ranges)

    def __getitem__(self, idx):
        # Check if current session dataset is loaded and if idx is within the current range
        if (self.current_session_dataset is not None and
                self.current_session_start_idx <= idx < self.current_session_start_idx + len(self.current_session_dataset)):
            return self.current_session_dataset[idx - self.current_session_start_idx]

        # Load the correct session dataset if not loaded or idx is out of the current range
        for session_path, (start_idx, end_idx) in zip(self.session_paths, self.index_ranges):
            if start_idx <= idx < end_idx:
                # print(f"subject-session from {session_path}")
                self.current_session_dataset = EEGSesDataset(session_path)
                self.current_session_start_idx = start_idx
                return self.current_session_dataset[idx - start_idx]

        raise IndexError("Index out of range")


class EEGDataset(Dataset):
    def __init__(self, subj_nums, root_path):
        self.subj_datasets = []

        self.index_ranges = []
        total_index = 0
        for num in subj_nums:
            subj_path = os.path.join(root_path, f'sub-{num:02}')  # Correctly format the subject directory name
            nSub = EEGSubjDataset(subj_path)
            self.subj_datasets.append(nSub)
            num_files = len(nSub)  # Get the number of valid files or entries in the dataset
            if num_files > 0:
                self.index_ranges.append((total_index, total_index + num_files))
                total_index += num_files

    def __len__(self):
        return sum(end - start for start, end in self.index_ranges)

    def __getitem__(self, idx):
        # Loop through all subject datasets to find the correct data index
   
        for subj_dataset, (start_idx, end_idx) in zip(self.subj_datasets, self.index_ranges):
            if start_idx <= idx < end_idx:
                return subj_dataset[idx - start_idx]

        raise IndexError("Index out of range")


# # Initialize dataset
# import random

# data_dir = '/Users/jeonsang-eon/sleep_data_processed/'
# subj_nums = [1, 2]
# ex = EEGDataset(subj_nums=subj_nums, root_path=data_dir)
# l = len(ex)
# lt = [i for i in range(l)]
# random.shuffle(lt)  # Correctly shuffle the list

# for i in lt:
#     data, label = ex[i]  # Assuming each item returns a tuple (data, label)
#     print(data.shape)
#     print(label)
#     if(data.shape==0):
#         print('Wrong')

# print(ex)
# print(ex[50])
# print(ex[50][0].shape)
# print(ex[50][1])
# print(len(ex))
# loader = DataLoader(dataset, batch_size=10, shuffle=True)
