import csv

# Replace 'your_delimiter' with the delimiter used in your text file
delimiter = '	'

with open('number_reading\\training\\labels.txt', 'r') as infile, open('number_reading\\train.csv', 'w', newline='') as outfile:
    reader = csv.reader(infile, delimiter=delimiter)
    writer = csv.writer(outfile)
    writer.writerow(['image', 'value'])


    for row in reader:
        # Assuming you want to split each line into two columns
        # Modify this part based on how you want to process each line
        if len(row) >= 2:
            writer.writerow([row[0], row[1]])
        else:
            # Handle lines that don't have enough values
            pass
