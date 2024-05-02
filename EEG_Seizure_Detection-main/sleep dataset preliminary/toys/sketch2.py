import mne
import matplotlib.pyplot as plt

# Load the EDF file
file_path = '/Users/jeonsang-eon/ds003768-download/sub-01/eeg/sub-01_task-rest_run-1_eeg_ref.edf'
raw = mne.io.read_raw_edf(file_path, preload=True)

# Print the list of all channels
print("List of all channels:")
print(raw.ch_names)

# # Plot the raw data
raw.plot()

plt.show()