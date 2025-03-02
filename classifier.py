import csv
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.classify import NaiveBayesClassifier
from collections import defaultdict
import random

nltk.download('punkt')
nltk.download('stopwords')

DATADIR = "./dataset"

#______________________________________DATA PROCESSING_____________________________________________
def read_data(filename):
    movies = []
    all_genres = set()
    
    with open(filename, "r", newline="", encoding="utf-8") as infile:
        reader = csv.reader(infile)
        next(reader)
        
        for row in reader:
            genres = [g.strip().lower() for g in row[0].split(",")]
            overview = row[1].lower()
            
            if genres and overview:
                movies.append({"genres": genres, "text": overview})
                all_genres.update(genres)
    
    return movies, sorted(all_genres)

movies, unique_genres = read_data(f"{DATADIR}/filtered_movie_dataset.csv")
print(f"Loaded {len(movies)} movies with {len(unique_genres)} unique genres")
