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
from sklearn.naive_bayes import ComplementNB

import nltk
from sklearn.naive_bayes import MultinomialNB
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
    max_features=10000,
    min_df=2,
    max_df=0.85,
    use_idf=True,
    norm='l2',
    #Captures single or word pairs
    ngram_range=(1,3)
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


# #______________________________________Models____________________________________________________
# In this section, a couple of classifiers and models have been added. 
# Trying to compare them eachother to grasp the effectiveness of each solution.
print("______________MODELS________________")

# using Multi-label kNN classifier with regression model
main_classifier = MultiOutputClassifier(
    LogisticRegression(max_iter=1000, class_weight='balanced'),
    n_jobs=-1
)


# using Multi-label kNN classifier with naivebayes model
#main_classifier = MultiOutputClassifier(ComplementNB(), n_jobs=-1)
#Problem with naivebayes, somehow receives zero division error

# OneVsRestClassifier
# main_classifier = OneVsRestClassifier(
#     LinearSVC(C=1.0, class_weight='balanced', max_iter=2000, dual=False)
# )
#This is quick!

# # OneVsRestClassifier with randomforest
# main_classifier = OneVsRestClassifier(
#     RandomForestClassifier(
#         n_estimators=100, 
#         max_depth=None, 
#         min_samples_split=2, 
#         min_samples_leaf=1,
#         class_weight='balanced'
#     )
# )
#VERY SLOW, 10 minutes and still can't print accuracy.


# #______________________________________TRAINING____________________________________________________
# In this section, the chosen solution is trained.


main_classifier.fit(X_train, y_train) 

#Testing the classifier on some sentence, expected label here is: animation
test_sentences = ["clip science please collection force water us archival footage animated illustration amusing narration explain archimedes principle thing float others sink."
]

test_sentences_tfidf = vetorizar.transform(test_sentences)

#This is a vector in binary form
predicted_labels = main_classifier.predict(test_sentences_tfidf)

#Inverses into tokens
print("______________TRAINING________________")
predicted_genres = mlb.inverse_transform(predicted_labels)
print("Predicted genres:", predicted_genres)
#print(len(main_classifier.classes_))
# print(predicted_labels)
# print("type of predicted_labels: ", type(predicted_labels))
# print("type of y_test: ", type(y_test[0]))
# print(mlb.inverse_transform(np.array([y_test[0]])))

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
    
    
print("______________EVALUATION_______________")
our_evaluate(main_classifier, list(y_test)[30000:31000], X_test[30000:31000])


# #______________________________________EVALUATION BUILT-IN__________________________________________________
# This section contains the built-in evaluation tools.


# predicted = main_classifier.predict(X_test)

# print(classification_report(y_test, predicted))


# # Checks for the evaluation metrics.
# #accuracy = accuracy_score(y_test, predicted)
# precision = precision_score(y_test, predicted, average='micro')
# recall = recall_score(y_test, predicted, average='micro')
# f1 = f1_score(y_test, predicted, average='micro')

# #print("Accuracy:", accuracy)
# print("Precision:", precision)
# print("Recall:", recall)
# print("F1 Score:", f1)



