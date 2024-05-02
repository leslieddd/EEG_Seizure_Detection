import os
import json
from prettytable import PrettyTable

labelFs = {}  # Dictionary to hold label counts

root_dir = '/Users/jeonsang-eon/sleep_data_processed/'

# Walk through the directory
for dirpath, _, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.endswith('.json'):  # Check if the file is a JSON file
            file_path = os.path.join(dirpath, filename)  # Full path to the file
            with open(file_path, 'r') as file:  # Open the file for reading
                data = json.load(file)  # Load the JSON data from the file
                label = data.get("label")  # Get the label value from the JSON data
                if label in labelFs:
                    labelFs[label] += 1  # Increment the count for the existing label
                else:
                    labelFs[label] = 1  # Initialize the count for the new label

# Set up the pretty table
table = PrettyTable()
table.field_names = ["Label", "Count"]  # Define the columns

# Fill the table with data
for label, count in labelFs.items():
    table.add_row([label, count])

print(table)  # Print the final counts of labels in a table format
