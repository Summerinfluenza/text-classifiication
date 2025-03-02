import csv
import os
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

filtered_file = "./filtered_movie_dataset.csv"
pre_processed_file = "pre_processed_dataset.csv"

#______________________________________DATA PROCESSING_____________________________________________

# Lemmatization of the text - define this function before using it
def preprocess(text):
    tokens = nltk.word_tokenize(text)
    return ' '.join([
        WordNetLemmatizer().lemmatize(t.lower()) 
        for t in tokens 
        if t.isalpha() and t not in stopwords.words('english')
    ])

def write_preprocessed_data(original_file, filename):
    # Check if the input file exists
    if not os.path.exists(original_file):
        print(f"Error: Input file '{original_file}' not found.")
        return
    
    try:
        with open(original_file, "r", encoding="utf-8") as infile, \
             open(filename, "w", newline="", encoding="utf-8") as outfile:
            
            reader = csv.reader(infile)
            writer = csv.writer(outfile)
            
            # Read header
            header = next(reader)
            # Write header to output file
            writer.writerow(["overview", "genres"])
            
            # Process each row
            row_count = 0
            for row in reader:
                try:
                    if len(row) >= 2:  # Ensure row has at least 2 columns
                        overview = row[0].strip().lower()
                        genre_str = row[1].strip().lower()
                        
                        # Filter rows with empty genre or overview
                        if genre_str and overview:
                            # Preprocess the overview
                            # Don't have to do it on genre_labels since it's already lemmatized
                            processed_overview = preprocess(overview)

                            # Get the genre list
                            genre_list = [g.strip() for g in genre_str.split(",")]
                            genre_str_output = ",".join(genre_list)
                            
                            # Write processed data
                            writer.writerow([processed_overview, genre_str_output])
                            row_count += 1
                except Exception as e:
                    print(f"Error processing row: {e}")
                    continue
            
            print(f"Successfully preprocessed {row_count} rows and saved to {filename}")
    except Exception as e:
        print(f"Error during preprocessing: {e}")

# Run the preprocessing
write_preprocessed_data(filtered_file, pre_processed_file)