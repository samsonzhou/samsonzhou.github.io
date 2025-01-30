import json
import csv

# Load the JSON file
with open("meta-test-dataset.json", "r") as file:
    data = json.load(file)

counts = set()

csv_data = list()

for model in data.keys():
    j = 0
    model_stats = list()
    for sub in data[model].keys():
        if j < 2:
            model_stats.append([data[model][sub]['y'][i][0] for i in range(200)])
            j += 1
    csv_data.append(model_stats)

# Writing to a TXT file, preserving original structure
with open("output.txt", "w") as file:
    for sublist in csv_data:
        file.write(repr(sublist) + "\n")  # Write each list-of-lists on a new line
