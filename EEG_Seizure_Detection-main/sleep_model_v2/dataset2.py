import os
import pdb

import numpy as np
import json
from torch.utils.data import Dataset, DataLoader
import torch


class EEGDataset(Dataset):
    def __init__(self, subj_nums, root_path):
        subj_dir_paths = [os.path.join(root_path, f"sub-{num:02d}") for num in subj_nums]
        self.index_ranges = []
        self.subjDataset = []
        total_index = 0
        for subj_dir_path in subj_dir_paths:
            subjDset = EEGSubjDataset(subj_dir_path)
            self.subjDataset.append(subjDset)
            self.index_ranges.append((total_index, total_index + len(subjDset)))
            total_index += len(subjDset)

    def __len__(self):
        return sum(end - start for start, end in self.index_ranges)

    def __getitem__(self, idx):
        for subjDset, (start_idx, end_idx) in zip(self.subjDataset, self.index_ranges):
            if start_idx <= idx < end_idx:
                return subjDset[idx - start_idx]
        raise IndexError("Index out of range")

class EEGSubjDataset(Dataset):
    def __init__(self, subj_path):
        # LABEL_DICT = {"W": 0, "S": 1}
        self.LABEL_DICT = {"W": 0, "S": 1, "1": 1, "2": 2}
        origin_files = os.listdir(subj_path)
        self.paths = []
        self.labels = []
        for f in origin_files:
            if f.endswith('.npy'):
                npy_path = os.path.join(subj_path, f)
                json_path = npy_path.replace('.npy', '.json')
                if not os.path.exists(json_path):
                    print(f'json file not found {json_path}')
                    continue
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    data['label'] = str(data['label'])
                    # print(json_path)
                    # pdb.set_trace()
                    self.labels.append(self.LABEL_DICT[data['label']])
                self.paths.append(npy_path)


# class EEGSubjDataset(Dataset):
#     def __init__(self, subj_path):
#         self.LABEL_DICT = {"W": 0, "S": 1, "1": 1, "2": 2}  # 假设 '1' 也是有效的标签
#         origin_files = os.listdir(subj_path)
#         self.paths = []
#         self.labels = []
#         for f in origin_files:
#             if f.endswith('.npy'):
#                 npy_path = os.path.join(subj_path, f)
#                 json_path = npy_path.replace('.npy', '.json')
#                 if not os.path.exists(json_path):
#                     print(f'JSON file not found for {npy_path}')
#                     continue
#                 with open(json_path, 'r') as f:
#                     data = json.load(f)
#                     label_str = str(data['label'])  # 将标签统一转为字符串
#                     if label_str in self.LABEL_DICT:
#                         self.labels.append(self.LABEL_DICT[label_str])
#                     else:
#                         print(f"Label '{label_str}' not recognized. Found in {json_path}")
#                         continue  # 如果标签不在字典中，不加载这个文件
#                 self.paths.append(npy_path)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        data = np.load(self.paths[idx]).astype(np.float32)  # Convert to float32
        data = torch.tensor(data)
        label = self.labels[idx]
        return data, label

# # Example usage
# data_dir = '/Users/jeonsang-eon/sleep_data_processed2/'
# subj_nums = [1, 2]
# dataset = EEGDataset(subj_nums=subj_nums, root_path=data_dir)
# print(f"Dataset Length: {len(dataset)}")
# l=len(dataset)
# for i in range(l):
#     print(dataset[0][0].shape)

