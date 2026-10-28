# WDM – Ex. No. 8: Text Classification

## Aim

To perform text classification on movie reviews using Python.

## Description

The following steps are required to create a text classification model in Python:

1. Importing Libraries
2. Importing the Dataset
3. Text Preprocessing
4. Converting Text to Numbers
5. Splitting into Training and Testing Sets
6. Training the Text Classification Model and Predicting Sentiment
7. Evaluating the Model
8. Saving and Loading the Model

## Program

### 1. Importing the Libraries and Dataset

```python
import numpy as np
import re
import nltk
import pickle

nltk.download("wordnet")
nltk.download("stopwords")

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_files
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

movie_data = load_files(r"/content/sample_data/txt_sentoken")

X, y = movie_data.data, movie_data.target

documents = []
stemmer = WordNetLemmatizer()
```

### 2. Text Preprocessing

```python
for sen in range(0, len(X)):
    document = re.sub(r"\W", " ", str(X[sen]))
    document = re.sub(r"\s+[a-zA-Z]\s+", " ", document)
    document = re.sub(r"\^[a-zA-Z]\s+", " ", document)
    document = re.sub(r"\s+", " ", document, flags=re.I)
    document = re.sub(r"^b\s+", "", document)
    document = document.lower()
    document = document.split()

    document = [
        stemmer.lemmatize(word)
        for word in document
    ]

    document = " ".join(document)
    documents.append(document)
```

### 3. Converting Text to Numbers

```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(
    max_features=1500,
    min_df=5,
    max_df=0.7,
    stop_words=stopwords.words("english")
)

X = vectorizer.fit_transform(documents).toarray()
```

### 4. Applying TF-IDF Transformation

```python
from sklearn.feature_extraction.text import TfidfTransformer

tfidfconverter = TfidfTransformer()

X = tfidfconverter.fit_transform(X).toarray()
```

### 5. Splitting the Dataset

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=0
)
```

### 6. Training the Random Forest Classifier

```python
classifier = RandomForestClassifier(
    n_estimators=1000,
    random_state=0
)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
```

### 7. Evaluating the Model

```python
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
```

## Dataset

The model uses the movie review dataset stored in the `txt_sentoken` directory. It contains positive and negative movie reviews used to train and test the classification model.

Example review:

```text
Plot: Two teen couples go to a church party, drink, and then drive.
They get into an accident. One of the guys dies, but his girlfriend
continues to see him in her life and has nightmares.

Critique: The movie has an interesting concept, but its confusing
storyline and lack of explanation make it less entertaining.
```

## Output

The program displays:

- Confusion matrix
- Classification report, including precision, recall, and F1-score
- Model accuracy

### Output Screenshot

![Text Classification Output](https://github.com/user-attachments/assets/a8f40236-31c0-4e1f-9ee5-ed68ff3bfd8e)

## Result

Thus, text classification was performed successfully using Python to predict the sentiment of movie reviews.
