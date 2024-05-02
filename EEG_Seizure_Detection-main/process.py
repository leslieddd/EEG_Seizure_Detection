from data_management import get_folder_info
from preprocessing import preprocess
import mne 
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy

files_list = get_folder_info()
freq_bands = {'delta':(0.5,4), 'theta':(4,8), 'alpha':(8,12), 'beta':(12,30), 'gamma':(30,120)}

# Preictal duration in minutes
# window_duration (in seconds) --> 30 s
# num_time_steps = 3 (in seconds)
# window_duration / num_time_steps should be an INTEGER! Otherwise process() will not work

def process(patient_num, preictal_duration, window_duration, num_time_steps):
    patient_folder = f'chb{patient_num:02d}'
    patient_file_list = sorted(list(files_list[patient_folder].keys()))

    # Contains label for each sample (inter, pre, ict)
    y_sample_labels = []

    raws = []
    fs = 256
    
    # Concatenate all labels for continuous data samples
    for patient_file in patient_file_list:
        raw, label_list = preprocess(patient_num, patient_file, preictal_duration)
        y_sample_labels.extend(label_list)
        raws.append(raw)
    raws_concatenated = mne.concatenate_raws(raws)

    # Ictal labeling
    start_indices = []
    start_index = None

    for i, item in enumerate(y_sample_labels):
        if item == 'ict' and (start_index is None or y_sample_labels[i-1] != 'ict'):
            start_index = i
            start_indices.append(i)

    y_sample_labels = np.array(y_sample_labels)

    # Preictal labeling (specified minutes before onset in preictal_duration)
    for start_index in start_indices:
        y_sample_labels[(start_index-fs*preictal_duration*60):start_index] = 'pre'

    # All other indices labeled as interictal
    y_sample_labels[y_sample_labels == None] = 'inter'

    #count_inter = np.count_nonzero(y == 'inter')
    #count_ict = np.count_nonzero(y_sample_labels == 'ict')
    #count_pre = np.count_nonzero(y == 'pre')

    #print(count_inter, count_ict, count_pre)

    sequence_duration = int(window_duration/num_time_steps)  # 10 sec (in example)
    #sequence_length = sequence_duration*fs
    sequences = mne.make_fixed_length_epochs(raws_concatenated, duration=sequence_duration)

    y_epoch_labels = []
    feature_lists = []

    #window_duration = 30  # Duration of each epoch in seconds
    windows = mne.make_fixed_length_epochs(raws_concatenated, duration=window_duration)
    window_samples = window_duration*fs

    for i, window in enumerate(windows):
        y_epoch_sample_labels = y_sample_labels[i*window_samples : (i+1)*window_samples]
        labels = ['inter', 'pre', 'ict']
        # Check whether samples in each epoch have same label or different ones 
        counts = {label: np.sum(y_epoch_sample_labels == label) for label in labels}
        y_epoch_labels.append(max(counts, key=counts.get))


    for sequence in sequences:
        psds, freqs = mne.time_frequency.psd_array_multitaper(sequence, sfreq=fs, fmin=0, fmax=120, adaptive=True)
        #feature_matrix = np.zeros((18, 15))

        feature_list = []

        for i, channel_psd in enumerate(psds):
            for j, band in enumerate(freq_bands):
                # Mean spectral power calculation
                fmin, fmax = freq_bands[band]
                freq_indices = np.where((freqs >= fmin) & (freqs < fmax))[0]
                band_psd = channel_psd[freq_indices]
                mean_spectral_power = np.mean(band_psd)

                # Spectral entropy calculation
                normalized_band_psd = band_psd / np.sum(band_psd)
                spectral_entropy = entropy(normalized_band_psd, base=10)

                # Mean spectral amplitude calculation
                epoch_data = sequence[i,:]
                N = len(epoch_data)
                fft_values = np.fft.fftshift(np.fft.rfft(epoch_data))
                fft_freqs = np.arange(-fs/2, fs/2, fs/N)

                fft_freq_indices = np.where((freqs >= fmin) & (freqs < fmax))[0]
                spectrum_amplitude = np.abs(fft_values[fft_freq_indices])
                mean_spectrum_amplitude = np.mean(spectrum_amplitude)

                band_feature_list = [mean_spectral_power, spectral_entropy, mean_spectrum_amplitude]
                feature_list.extend(band_feature_list)

        #feature_matrices.append(feature_matrix)
        feature_lists.append(feature_list)
        
    feature_lists = np.array(feature_lists)
    mean = np.mean(feature_lists, axis=0)
    std = np.std(feature_lists, axis=0)

    # Perform standard normalization
    normalized_feature_lists = (feature_lists - mean) / std

    x_dim = int(len(feature_lists)/num_time_steps)
    
    x = normalized_feature_lists.reshape(x_dim, num_time_steps, 15*18)
    print(x)
    #print(x.shape)
    #print(y_epoch_labels)

    # Create one-hot encoded y output (input to model)
    mapping = {'pre': [1, 0, 0], 'inter': [0, 1, 0], 'ict': [0, 0, 1]}

    # Convert strings to one-hot encoded array
    y = np.array([mapping[y_epoch_label] for y_epoch_label in y_epoch_labels])

    return x, y

# Test the function below!
# process(16)
