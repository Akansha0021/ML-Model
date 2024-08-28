# Twitter Sentiment Analysis using Machine Learning

# Installing the Kaggle library
!pip install kaggle

# Upload your Kaggle.json file

# Configuring the path of the Kaggle.json file
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Importing the Twitter Sentiment dataset
# API to fetch the dataset from Kaggle
!kaggle datasets download -d kazanova/sentiment140

# Extracting the compressed dataset
from zipfile import ZipFile
dataset = 'sentiment140.zip'

with ZipFile(dataset, 'r') as zip_ref:
    zip_ref.extractall()
    print('The dataset is extracted')

# Importing the dependencies
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Downloading stopwords from NLTK
nltk.download('stopwords')

# Printing the stopwords in English
print(stopwords.words('english'))

# Data Processing
# Loading the data from CSV file to pandas dataframe
twitter_data = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='ISO-8859-1', header=None)

# Checking the number of rows and columns
print(twitter_data.shape)

# Printing the first 5 rows of the dataframe
print(twitter_data.head())

# Counting the number of missing values in the dataset
print(twitter_data.isnull().sum())

# Checking the distribution of the target column
print(twitter_data[0].value_counts())

# Convert the target "4" to "1"
twitter_data.replace(4, 1, inplace=True)

# Checking the distribution of the target column after replacement
print(twitter_data[0].value_counts())

# Stemming the tweets
port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

# Applying the stemming function to the tweet column
twitter_data['tweet'] = twitter_data[5].apply(stemming)

print(twitter_data['tweet'].head())

# Separating the data and label
X = twitter_data['tweet'].values
Y = twitter_data[0].values

print(X[:5])
print(Y[:5])

# Splitting the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

# Converting the textual data to numerical data using TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(X)

X_train = vectorizer.transform(X_train)
X_test = vectorizer.transform(X_test)

print(X_train.shape)
print(X_test.shape)

# Training the Machine Learning Model using Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

# Model Evaluation
# Accuracy score on the training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)

print('Accuracy score of the training data:', training_data_accuracy)

# Accuracy score on the test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)

print('Accuracy score of the test data:', test_data_accuracy)

# Saving the trained model
import pickle

filename = 'trained_model.sav'
pickle.dump(model, open(filename, 'wb'))

# Using the saved model for future predictions
# Loading the saved model
loaded_model = pickle.load(open(filename, 'rb'))

# Testing with new data
X_new = X_test[200]
Y_new = Y_test[200]

prediction = loaded_model.predict(X_new)
print('Actual label:', Y_new)
print('Predicted label:', prediction[0])

if prediction[0] == 0:
    print('Negative Tweet')
else:
    print('Positive Tweet')

# Another test
X_new = X_test[3]
Y_new = Y_test[3]

prediction = loaded_model.predict(X_new)
print('Actual label:', Y_new)
print('Predicted label:', prediction[0])

if prediction[0] == 0:
    print('Negative Tweet')
else:
    print('Positive Tweet')
