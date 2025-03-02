import csv
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, hamming_loss
from sklearn.preprocessing import MultiLabelBinarizer
import nltk
from nltk.corpus import stopwords
import numpy as np

DATADIR = "./dataset"

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

#______________________________________FEATURE ENGINEERING__________________________________________

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


# #______________________________________TRAINING____________________________________________________

# using Multi-label kNN classifier 
mlknn_classifier = MultiOutputClassifier(
    LogisticRegression(max_iter=1000, class_weight='balanced'),
    n_jobs=-1
)
mlknn_classifier.fit(X_train, y_train) 

#Testing the classifier on some sentence, expected label here is: animation
test_sentences = ["clip science please collection force water us archival footage animated illustration amusing narration explain archimedes principle thing float others sink."
]

test_sentences_tfidf = vetorizar.transform(test_sentences)

#This is a vector in binary form
predicted_labels = mlknn_classifier.predict(test_sentences_tfidf)

#Inverses into tokens
print("______________TRAINING________________")
predicted_genres = mlb.inverse_transform(predicted_labels)
print("Predicted genres:", predicted_genres)
# print(predicted_labels)
# print("type of predicted_labels: ", type(predicted_labels))
# print("type of y_test: ", type(y_test[0]))
# print(mlb.inverse_transform(np.array([y_test[0]])))

# #______________________________________EVALUATION__________________________________________________

def accuracy(classifier, genres_list, overviews):

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
    
    
print("______________EVALUATION_______________")
our_evaluate(mlknn_classifier, list(y_test)[20000:21000], X_test[20000:21000])