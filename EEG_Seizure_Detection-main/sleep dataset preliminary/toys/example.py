import mne

# Assuming the data has been loaded successfully as per your instructions
file_path = './test/sub-01_task-rest_run-1_eeg'
raw = mne.io.read_raw_brainvision(file_path + '.vhdr', preload=True)

# Downsample the data to 256 Hz
raw.resample(256)

# Define the bipolar montage using correct anode-cathode pairs
bipolar_pairs = [
    ('Fp1', 'F7'), ('F7', 'T7'), ('T7', 'P7'), ('P7', 'O1'),
    ('Fp1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1'),
    ('Fp2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2'),
    ('Fp2', 'F8'), ('F8', 'T8'), ('T8', 'P8'), ('P8', 'O2'),
    ('Fz', 'Cz'), ('Cz', 'Pz')
]

# Apply bipolar referencing
for anode, cathode in bipolar_pairs:
    mne.set_bipolar_reference(raw, anode=[anode], cathode=[cathode], drop_refs=False, copy=False)

# Drop original channels (only if they are no longer needed)
raw.drop_channels(['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'Oz', 'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'TP9', 'TP10', 'EOG', 'ECG'])

# Specify the output file path for the EDF file
output_file_path = './modified_data.edf'

# Use the export_raw function to save the Raw object as an EDF file
mne.export.export_raw(output_file_path, raw, fmt='EDF', overwrite=True)

