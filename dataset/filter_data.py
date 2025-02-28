import csv

raw_data_file = "./movie_dataset.csv"
filtered_file = "filtered_movie_dataset.csv"

with open(raw_data_file, "r", newline="", encoding="utf-8") as infile, \
     open(filtered_file, "w", newline="", encoding="utf-8") as outfile:

    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    for row in reader:
        if len(row) >= 17:
            writer.writerow([row[13], row[16]])