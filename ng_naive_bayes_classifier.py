import pandas as pd
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Step 1: Cleaning the dataset for training
filename = "pre_processed_dataset.csv"

# Create a dictionary for genres and overviews
genres = {}
overviews = {}
overviews["overview"] = []

# Loop through the dataset
with open(filename, "r", newline="", encoding="utf-8") as infile:
    reader = csv.reader(infile)
    count = 0

    # Checks each row for overview and genres
    for row in reader:
        if (count > 500000):
            break
        overviews["overview"].append(row[0])
        curr_genres = row[1].split(",")

        # Add new list if genre seen for the first time
        for genre in curr_genres:
            if genre not in genres:
                genres[genre] = {}
                genres[genre]["label"] = ['0'] * count

        # Append 1 if genre in current list, else 0 if not
        for genre in genres:
            if genre in curr_genres:
                genres[genre]["label"].append(1)
            else:
                genres[genre]["label"].append(0)
        
        count += 1

# Create a Naive Bayes Classifier for each genre
model_list = {}
for genre in genres:
    genres[genre].update(overviews)
    df = pd.DataFrame(genres[genre])
    
    # Split the data into 70% train and 30% test
    x = df['overview']
    y = df['label']
    y = y.astype('int') 
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # Vectorize the Text Data
    vectorizer = CountVectorizer()
    x_train_vectors = vectorizer.fit_transform(x_train)
    x_test_vectors = vectorizer.transform(x_test)

    # Train the Naive Bayes model
    model = MultinomialNB()
    model.fit(x_train_vectors, y_train)
    model_list[genre] = model

    # Make predictions and evaluate accuracy
    y_pred = model.predict(x_test_vectors)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for {genre}: {accuracy * 100:.2f}%\n")

# Do some testing
print("Predicted Genres = ", end = '')
for genre in model_list:
    model = model_list[genre]
    message = ["story focus kitarou birth star kitarou father later medama oyaji mizuki kitarou foster parent"]
    vector = vectorizer.transform(message)
    prediction = model.predict(vector)
    if (prediction[0] == 1):
        print(genre + ",", end = '')

