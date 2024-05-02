def print_channel_names(vhdr_file_path):
    channel_info = False
    channels = []

    try:
        with open(vhdr_file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line == '[Channel Infos]':  # Start of channel info section
                    channel_info = True
                elif line.startswith('[') and channel_info:  # End of channel info section
                    break
                elif channel_info and '=' in line:
                    # Extract channel name, typically formatted as:
                    # Ch<index>=<label>,<reference point>,<resolution>,<unit>
                    channel_name = line.split('=')[1].split(',')[0]  # Extract the name part
                    channels.append(channel_name)
    except FileNotFoundError:
        print("File not found. Please check the file path.")
    except Exception as e:
        print(f"An error occurred: {e}")

    if channels:
        print("Channel Names:")
        for channel in channels:
            print(channel)
    else:
        print("No channel names found or file could not be read.")

# Example usage:
vhdr_file_path = '/Users/jeonsang-eon/sleep_data/sub-01/sub-01_task-rest_run-1_eeg.vhdr'
print_channel_names(vhdr_file_path)


