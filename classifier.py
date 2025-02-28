import csv


DATADIR = "./dataset"

def read_data(filename):
    movies = []

    with open(filename, "r", newline="", encoding="utf-8") as infile:

        reader = csv.reader(infile)

        #Checks for all movies from the filtered dataset
        for row in reader:
            movie = []
            #Removes any movie without genre or overview
            if not (row[1] == "" or row[0] == ""):
                movie.append(row[1])
                movie.append(row[0])
                movies.append(movie)
            
    return movies


all_movies = read_data(f"{DATADIR}/filtered_movie_dataset.csv")


#print(all_movies)
print(f"Appropriately tagged movies in this dataset: {len(all_movies)}")