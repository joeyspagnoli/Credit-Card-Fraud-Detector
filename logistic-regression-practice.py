import time
import polars as pl
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import random


#Data collection & pre-processing
"""
I want to test out using polars instead of pandas. I am going to create two functions that both read the email.csv
One of the functions will utilize pandas, the other polars. I will be tracking time to run each function and compare
"""

def pandas_csv():
    df = pd.read_csv('email.csv')
    return df

def polars_csv():
    df = pl.read_csv('email.csv')
    return df

def runtime():
    start = time.perf_counter()
    pandas_csv()
    elapsed_pd = time.perf_counter() - start
    print(f"pandas_csv took {elapsed_pd:.4f} seconds")

    start = time.perf_counter()
    polars_csv()
    elapsed_pl = time.perf_counter() - start
    print(f"polars_csv took {elapsed_pl:.4f} seconds")

    speed_up = elapsed_pd / elapsed_pl
    print(f"polars was {speed_up:.4f}x faster than pandas")

#runtime()
"""
pandas_csv took 0.0199 seconds
polars_csv took 0.0054 seconds
polars was 3.6714x faster than pandas
"""

df = pl.read_csv('email.csv')

# print(df.head())
#
# random.seed(42)
#
# print(df.sample(4))


#One-hot encode the category column spam - 1, ham - 0

df = df.with_columns( #Applies the changes
    pl.when(pl.col("Category") == "spam") #Checks each row in Category if it is equal to spam
    .then(1) #If it is spam, update its value to 1
    .otherwise(0) #Everything else is turned to 0
    .cast(pl.Int64) #Makes sure its the integer type
    .alias("Category") #Tells polars to overwite the existing Category column
)
print(df.head())

#Split into X and Y
X = df['Message']
Y = df['Category']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

#Build the pipeline
pipeline = Pipeline([
    #First thing that is needed to be done is tokenize X
    ('tfidf', TfidfVectorizer(stop_words='english')),

    #Then apply logistic regression to tokenized input
    ('clf', LogisticRegression())
])

model = pipeline.fit(X_train, Y_train)

preds = model.predict(X_test)

print("Accuracy:", accuracy_score(Y_test, preds))

test_data = list(zip(X_test.to_list(), Y_test.to_list()))

samples = random.sample(test_data, 5)
print("Ham : 0, Spam : 1")
print("\nSample Predictions:\n")
for message, true_label in samples:
    prediction = model.predict([message])[0]
    print(f"Message: {message[:60]}...")  # Truncate long messages
    print(f"True Label: {true_label}, Predicted: {prediction}")
    print("-" * 60)

input_mail = ["I've been searching for the right words to thank you for this breather. I promise i wont take your help for granted and will fulfil my promise. You have been wonderful and a blessing at all times"]

prediction = model.predict(input_mail)

print(prediction)

if (prediction[0]==0):
  print('Ham mail')

else:
  print('Spam mail')