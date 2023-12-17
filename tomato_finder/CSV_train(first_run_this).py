import os
import csv

# Define the directory containing the images
reject_directory = 'data\\Three Classes\\Reject'
accepted_directory1 = 'data\\Three Classes\\Unripe'
accepted_directory2 = 'data\\Three Classes\\Ripe'

# List of image extensions

# Create a CSV file and write the image names and condition
csv_filename = 'data\\train.csv'
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['name', 'condition'])  # Write header

    for filename in os.listdir(reject_directory):
        full_name = reject_directory + "\\" + filename
        writer.writerow([full_name, 'bad'])

    for filename in os.listdir(accepted_directory1):
        full_name = accepted_directory1 + "\\" + filename
        writer.writerow([full_name, 'good'])

    for filename in os.listdir(accepted_directory2):
        full_name = accepted_directory2 + "\\" + filename
        writer.writerow([full_name, 'good'])

print(f"CSV file '{csv_filename}' created with image names and conditions.")
