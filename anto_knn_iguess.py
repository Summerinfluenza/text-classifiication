import csv
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import numpy as np

#the idea is to use a KNN classifier to predict the genre of a movie based on its overview

DATADIR = "./dataset"

#first we will put the datas in a two entry array, one for the overview and one for the genres
def read_data(filename):
    genres = []
    overviews = []
    
    with open(filename, "r", newline="", encoding="utf-8") as infile:
        reader = csv.reader(infile)
        #Checks each row for overview and genres
        for row in reader:
            genres.append(row[1].split(","))
            overviews.append(row[0])
    return overviews, genres

# Processed dataset, lemmatzed overview
overviews, genre_labels = read_data(f"{DATADIR}/pre_processed_dataset.csv")

print(f"Total clean movies with genres: {len(overviews)}")
print(overviews[0], genre_labels[0])

#the code above is from Yi

#we will compare each words of the overviews and calculate the distance between them and the genre of the movie
#then do an average of the distance to determine the genre of the movie

# Load the model
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
#hopefully there is some cinematographic press articles in it
#I chose pretrained vectors because I don't have enough space to train my own

def DistinctGenres(genres):
    distinctGenres = []
    for genre in genres:
        if genre not in distinctGenres:
            distinctGenres.append(genre)
    return distinctGenres


#this function will calculate the distance between the overview and the genre
def similarityOverviewGenre(overview, distinctGenres):
    distance = 0
    for word in overview:
        distance += model.wmdistance(word, distinctGenres)
    return distance/len(overview)

#this function will calculate the distance between the overview and all the genres
def similarityOverviewGenres(overview, distinctGenres):
    distances = []
    for genre in distinctGenres:
        distances.append(similarityOverviewGenre(overview, genre))
    return distances

#this function will predict the genre of the movie
def predictGenre(overview, distinctGenres):
    distances = similarityOverviewGenres(overview, distinctGenres)
    return distinctGenres[np.argmin(distances)]

#this function will predict the genre of all the movies
def predictGenres(overviews, distinctGenres):
    predictions = []
    for overview in overviews:
        predictions.append(predictGenre(overview, distinctGenres))
    return predictions

#training
distinctGenres = DistinctGenres(genre_labels)
predictions = predictGenres(overviews, distinctGenres)


#Transforms data into termfrequency inverse document frequency representation
vetorizar = TfidfVectorizer(
    #tokenizer=nltk.word_tokenize,
    #Removes useless common words that proviide no value like: "the", "and", etc.
    stop_words=stopwords.words('english'),
    #Keep only the 5000 most common words, for the same of simplicity and prestanda
    max_features=5000,
    #Captures siingle or word pairs
    ngram_range=(1,2)
)

#Fitting the tf-idf vectors with overviews by:
#1. Building a vocabulary (determined by max_features=5000) from all movie overviews
#2. Calculates IDF for each term
X = vetorizar.fit_transform(overviews)

mlb = MultiLabelBinarizer()
# Coverts genre list into a binary(0, 1) matrix
y = mlb.fit_transform(genre_labels)


#Spliits dataset into training and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42) 

#evaluations 
#thinking abt copy-pasting Yi's code to evaluate the accuracy of the model
def accuracy(classifier, genres_list, overviews): #classifier is the model, genres_list is the true genres, overviews is the overviews

    # Counts from 0
    documents = 0
    true_positives = 0

    #Loops all documents
    for genres, overview in zip(genres_list, overviews):
        
        #For each document, check if the true class matches the predicted class
        predicted_genres = mlb.inverse_transform(classifier.predict(overview))[0]
        true_genres = mlb.inverse_transform(np.array([genres]))[0]

        #print(set(true_genres), set(predicted_genres))
        #To check if all true_genres exist in predicted_genres
        # result = set(true_genres).issubset(set(predicted_genres))

        #For the sake of simplicity, predicting one correct genre is enough
        if set(true_genres).intersection(set(predicted_genres)):
            #Counts each correctly predicted document
            true_positives += 1
        documents += 1

    #To avoid zero division
    if documents == 0:
        return 0
    else:
        return true_positives / documents

def precision(classifier, genre, genres_list, overviews):
    # Counts from 0
    true_positives = 0
    false_positives = 0

    #Loops all documents
    for genres, overview in zip(genres_list, overviews):

        #For each document, check if the true class matches the predicted class
        predicted_genres = mlb.inverse_transform(classifier.predict(overview))[0]
        true_genres = mlb.inverse_transform(np.array([genres]))[0]
        
        #Checks if the predicted_class matches the given class

        # Mtching genres with predicted and true genres is true positive, rest is false posivive
        if set(genre).intersection(set(predicted_genres)):
            if set(genre).intersection(set(true_genres)):
                true_positives += 1
            else:
                false_positives += 1
                print(set(genre), set(true_genres))
            

    #To avoid zero division
    if (true_positives + false_positives) == 0:
        return 0
    else:
        return true_positives / (true_positives + false_positives)

def recall(classifier, c, overviews):
    """Compute the class-specific recall of a classifier on a list of gold-standard overviews."""
    # TODO: Implement this method to solve Problem 2
    # Counts from 0
    true_positives = 0
    false_negatives = 0

    #Loops all documents
    for correct_class, sample in overviews:
        predicted_class = classifier.predict(sample)
        
        #Checks if the predicted_class matches the given class
        if predicted_class == c and correct_class == c:
            #Counts each correctly predicted document
            true_positives += 1
        elif predicted_class != c and correct_class == c:
            false_negatives += 1
            
    #To avoid zero division
    if (true_positives + false_negatives) == 0:
        return 0
    else:
        return true_positives / (true_positives + false_negatives)

def our_evaluate(classifier, genres_list, overviews):
    #print("accuracy = {:.2%}".format(accuracy(classifier, genres_list, overviews)))
    for c in mlb.classes_:
        p = precision(classifier, [c], genres_list, overviews)
        print("class {}: precision = {:.2%}".format(c, p))
    #     r = recall(classifier, c, overviews)
    #     # TODO: Change the next line to compute the F1-score
    #     f = 2 * p * r/ (p + r)
    #     print("class {}: precision = {:.2%}, recall = {:.2%}, f1 = {:.2%}".format(c, p, r, f))
    