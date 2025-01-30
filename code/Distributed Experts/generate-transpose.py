import csv

# Read CSV and transpose it
with open("matrix.csv", "r") as infile:
    reader = list(csv.reader(infile))  # Convert CSV to list of lists
    transposed_data = list(zip(*reader))  # Transpose using zip()

# Write the transposed data back to CSV
with open("transposed.csv", "w", newline="") as outfile:
    writer = csv.writer(outfile)
    writer.writerows(transposed_data)

print("CSV file transposed successfully!")
