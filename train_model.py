
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv(
    "dataset/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)

# Separate messages and labels
X = data["message"]
y = data["label"]

# Divide data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create a machine learning pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

# Train the model
model.fit(X_train, y_train)

# Predict test messages
predictions = model.predict(X_test)

from sklearn.metrics import confusion_matrix

# Generate confusion matrix
cm = confusion_matrix(
    y_test,
    predictions,
    labels=["ham", "spam"]
)

print("\nConfusion Matrix:")
print(cm)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)

print("Model Training Completed!")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))



# Test with custom messages

test_messages = [
    "Congratulations! You won a free prize. Claim now!",
    "Hey, are you coming to college tomorrow?",
    "URGENT! You have won a cash reward. Call now!",
    "Please send me the assignment notes."
]

# Get spam probabilities
custom_probabilities = model.predict_proba(test_messages)

spam_index = list(model.classes_).index("spam")

print("\nCustom Message Predictions:")

for message, probability in zip(test_messages, custom_probabilities):

    spam_probability = probability[spam_index]

    if spam_probability >= 0.3:
        prediction = "SPAM"
    else:
        prediction = "HAM"

    print("\nMessage:", message)
    print("Prediction:", prediction)
    print("Spam Probability:", round(spam_probability * 100, 2), "%")



import joblib

# Save the trained model
joblib.dump(model, "spam_model.pkl")

print("\nModel saved successfully!")