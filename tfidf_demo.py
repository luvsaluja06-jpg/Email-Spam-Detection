
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Read the dataset
data = pd.read_csv(
    "dataset/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Convert messages into numerical values
tfidf_matrix = vectorizer.fit_transform(data["message"])

# Display results
print("Original messages:")
print(data["message"].head())

print("\nTotal messages:", tfidf_matrix.shape[0])
print("Total unique words:", tfidf_matrix.shape[1])

print("\nFirst 15 words:")
print(vectorizer.get_feature_names_out()[:15])

# Display numerical values for first 5 messages
print("\nTF-IDF numerical representation:")
print(tfidf_matrix[:5].toarray())

