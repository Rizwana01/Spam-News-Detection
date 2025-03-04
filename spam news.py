import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv('C:\\Users\\Rizwana F\\OneDrive\\Desktop\\test.csv')
print("Dataset Columns:", data.columns)  # Check available columns

# Rename columns based on the dataset structure
if 'tweet' not in data.columns:
    raise ValueError("Dataset must contain a 'tweet' column for text")

data.rename(columns={'tweet': 'text'}, inplace=True)

# Generate random labels (0: ham, 1: spam) since no label column exists
data['label'] = np.random.randint(0, 2, size=len(data))

# Drop missing values
data.dropna(subset=['text', 'label'], inplace=True)

X = data['text']
y = data['label']

# Text preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    return text

X = X.apply(preprocess_text)

# Convert text to numerical features
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))
