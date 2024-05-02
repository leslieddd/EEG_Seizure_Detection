import os
import json
import numpy as np

ORDERED_CHANNEL_PAIRS = [
    ['Fp1', 'F3'], ['Fp2', 'F4'],
    ['P3', 'O1'], ['P4', 'O2']
]

LABEL_TRANS = {
    "W": "W",
    "1": "S",
    "2": "S",
    "3": "S"
}

src_path = '/Users/jeonsang-eon/sleep_data_processed/'
dst_path = '/Users/jeonsang-eon/sleep_data_processed2/'

for dirpath, dirnames, filenames in os.walk(src_path):
    for filename in filenames:
        if filename.endswith('.json'):
            json_path = os.path.join(dirpath, filename)
            npy_path = os.path.join(dirpath, filename.replace('.json', '.npy'))

            if not os.path.exists(npy_path):
                print(f'npy file not found {npy_path}')
                continue

            temp = dirpath.split('/')
            new_name = temp[-1] + '_' + filename.split('.')[0]  # [session]_[time]
            new_dst_path = os.path.join(dst_path,temp[-2])

            if not os.path.exists(new_dst_path):
                os.makedirs(new_dst_path)

            with open(json_path, 'r') as f:
                data = json.load(f)
                label = data['label']
                if label not in ["W", "1", "2", "3"]:
                    print('label not in ["W", "1", "2", "3"]')
                    print(json_path)
                    continue
                label = LABEL_TRANS[label]

                channels = data['channels']
                try:
                    idxs = [channels.index(pair) for pair in ORDERED_CHANNEL_PAIRS]
                except ValueError as e:
                    print(f"Error: {e} in file {json_path}")
                    continue

            data = np.load(npy_path)
            modified_data = [data[idx] for idx in idxs]
            modified_data = np.array(modified_data)
            np.save(os.path.join(new_dst_path, new_name + '.npy'), modified_data)  # Corrected path
            modified_meta_data = {
                'channels': ORDERED_CHANNEL_PAIRS,
                'label': label
            }
            with open(os.path.join(new_dst_path, new_name + '.json'), 'w') as f:
                json.dump(modified_meta_data, f)
