import csv
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC

from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, hamming_loss

import nltk
from sklearn.naive_bayes import MultinomialNB, ComplementNB, GaussianNB, BernoulliNB

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
# In this section, the data is being filtered and put into vectors. 
# Preparing data for the training and testing.


#Transforms data into termfrequency inverse document frequency representation
vetorizar = TfidfVectorizer(
    #tokenizer=nltk.word_tokenize,
    #Removes useless common words that proviide no value like: "the", "and", etc.
    stop_words=stopwords.words('english'),
    #Keep only the 10000 most common words, for the sake of simplicity and prestanda
    max_features=5000,
    min_df=3,
    max_df=0.90,
    use_idf=True,
    norm='l2',
    #Captures single or word pairs
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


# #______________________________________ALGORITHMS____________________________________________________
# In this section, using the same strategy with knn but with different algorithms.
print("______________Algorithms________________")
all_classifiers = []

#"______________Logistic regression________________")
LR_knn_classifier = MultiOutputClassifier(
    LogisticRegression(max_iter=2000, class_weight='balanced'),
    n_jobs=-1
)
all_classifiers.append(LR_knn_classifier)

#"______________Naive Bayes________________")
CNB_knn_classifier = MultiOutputClassifier(ComplementNB(), n_jobs=-1)
all_classifiers.append(CNB_knn_classifier)

BNB_knn_classifier = MultiOutputClassifier(BernoulliNB(), n_jobs=-1)
all_classifiers.append(BNB_knn_classifier)

#"______________LinearSVC________________")
LSVC_knn_classifier = MultiOutputClassifier(
    LinearSVC(C=1.0, class_weight='balanced', max_iter=2000, dual=False)
)
all_classifiers.append(LSVC_knn_classifier)


# #______________________________________TRAINING____________________________________________________
# In this section, the chosen solution is trained.
print("______________TRAINING________________")

for classifier in all_classifiers:
    #Trains the classifier
    classifier.fit(X_train, y_train) 

    #Testing the classifier on some sentence, expected label here is: animation
    # test_sentences = ["clip science please collection force water us archival footage animated illustration amusing narration explain archimedes principle thing float others sink."
    # ]

    # test_sentences_tfidf = vetorizar.transform(test_sentences)

    # #This is a vector in binary form
    # predicted_labels = classifier.predict(test_sentences_tfidf)

    # #Inverses into tokens
    
    # predicted_genres = mlb.inverse_transform(predicted_labels)
    # print("Predicted genres:", predicted_genres)


# #______________________________________EVALUATION__________________________________________________
# This section contains the self-built evaluation methods. 
# The metrics are artbitrary set and we should decide the standard to use.


def accuracy(classifier, genres_list, overviews):

    # Counts from 0
    overview_count = 0
    true_positives = 0

    #Loops all documents
    for genres, overview in zip(genres_list, overviews):
        
        #For each document, check if the true class matches the predicted class
        #predicted_genres = mlb.inverse_transform(classifier.predict(overview))[0]
        predicted_genres = mlb.inverse_transform(classifier.predict(overview.reshape(1, -1)))[0]

        true_genres = mlb.inverse_transform(np.array([genres]))[0]

        #print(set(true_genres), set(predicted_genres))
        #To check if all true_genres exist in predicted_genres
        # result = set(true_genres).issubset(set(predicted_genres))

        #For the sake of simplicity, predicting one correct genre is enough
        if set(true_genres).intersection(set(predicted_genres)):
            #Counts each correctly predicted document
            true_positives += 1
        overview_count += 1

    #To avoid zero division
    if overview_count == 0:
        return 0
    else:
        return true_positives / overview_count

def precision(classifier, genre, genres_list, overviews):
    # Counts from 0
    true_positives = 0
    false_positives = 0

    #Loops all documents
    for genres, overview in zip(genres_list, overviews):

        #For each document, check if the true class matches the predicted class
        predicted_genres = mlb.inverse_transform(classifier.predict(overview))[0]
        #predicted_genres = mlb.inverse_transform(classifier.predict(overview.reshape(1, -1)))[0]
        true_genres = mlb.inverse_transform(np.array([genres]))[0]
        
        #Checks if the predicted_class matches the given class

        # Matching genres with predicted and true genres is true positive, rest is false posivive
        if set(genre).intersection(set(true_genres)):
            if set(genre).intersection(set(predicted_genres)):
                true_positives += 1
            else:
                false_positives += 1
                print(set(genre), set(predicted_genres))
            

    #To avoid zero division
    if (true_positives + false_positives) == 0:
        return 0
    else:
        return true_positives / (true_positives + false_positives)

def recall(classifier, genre, genres_list, overviews):
    # Counts from 0
    true_positives = 0
    false_negatives = 0

    #Loops all documents
    for genres, overview in zip(genres_list, overviews):
        predicted_genres = mlb.inverse_transform(classifier.predict(overview))[0]
        #predicted_genres = mlb.inverse_transform(classifier.predict(overview.reshape(1, -1)))[0]
        true_genres = mlb.inverse_transform(np.array([genres]))[0]
        
        #Checks if the predicted_class matches the given class

        # Matching genres with predicted and true genres is true positive, rest is false negative
        if set(genre).intersection(set(predicted_genres)):
            if set(genre).intersection(set(true_genres)):
                true_positives += 1
            else:
                false_negatives += 1
                #print(set(genre), set(true_genres))
            
    #To avoid zero division
    if (true_positives + false_negatives) == 0:
        return 0
    else:
        return true_positives / (true_positives + false_negatives)


# #______________________________________BASELINE__________________________________________________
# This section contains the baseline, which uses the most frequent class.

def mfc_baseline(genres_list):
    genre_counter = {}

    #Creates a dictionary of genre and count
    for genres in genres_list:
        true_genres = mlb.inverse_transform(np.array([genres]))[0]
        for genre in true_genres:
            if genre in genre_counter:
                genre_counter[genre] += 1
            else:
                genre_counter[genre] = 1

    #Sorts and choses the most frequent genre
    sorted_genre_counter = sorted(genre_counter.items(), key=lambda kv:(-kv[1], kv[0]))

    #Returns the counter / total of overviews = baseline
    return sorted_genre_counter[0][1] / len(genres_list)

print("Baseline: {:.2%}".format(mfc_baseline(list(y_test))))


# #______________________________________RESULTS__________________________________________________
def our_evaluate(classifier, genres_list, overviews):
    #print("accuracy = {:.2%}".format(accuracy(classifier, genres_list, overviews)))
    precision_ratio = 0
    recall_ratio = 0
    for c in mlb.classes_:
        precision_ratio += precision(classifier, [c], genres_list, overviews)
        
        recall_ratio += recall(classifier, [c], genres_list, overviews)
    

    precision_ratio = precision_ratio / len(classifier.classes_)
    recall_ratio = recall_ratio / len(classifier.classes_)

    fmeasure = 2 * precision_ratio * recall_ratio/ (precision_ratio + recall_ratio)
    print("precision = {:.2%}, recall = {:.2%}, f1 = {:.2%}".format(precision_ratio, recall_ratio, fmeasure))
    

# #______________________________________EVALUATION BUILT-IN__________________________________________________
# This section contains the built-in evaluation tools.
print("______________EVALUATION_______________")

for classifier in all_classifiers:
    predicted = classifier.predict(X_test)
    print(f"Evaluation for classifier: {classifier}")
    print(classification_report(y_test, predicted))

#our_evaluate(main_classifier, list(y_test)[30000:31000], X_test[30000:31000])

