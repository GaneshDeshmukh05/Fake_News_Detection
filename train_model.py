import pandas as pd
import numpy as np
import pickle
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

print("Loading datasets...")

# TEXT CLEANING FUNCTION
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

# LOAD ISOT DATASET
fake_isot = pd.read_csv("dataset/Fake.csv")
true_isot = pd.read_csv("dataset/True.csv")

fake_isot["label"] = 0
true_isot["label"] = 1

fake_isot["text"] = fake_isot["title"] + " " + fake_isot["text"]
true_isot["text"] = true_isot["title"] + " " + true_isot["text"]

fake_isot = fake_isot[["text","label"]]
true_isot = true_isot[["text","label"]]

isot_data = pd.concat([fake_isot,true_isot])

print("ISOT dataset loaded")


# LOAD WELFAKE DATASET
welfake = pd.read_csv("dataset/WELFake_Dataset.csv")

# reverse labels if needed
welfake["label"] = welfake["label"].apply(lambda x: 0 if x==1 else 1)

welfake["text"] = welfake["title"].astype(str) + " " + welfake["text"].astype(str)

welfake = welfake[["text","label"]]

print("WELFake dataset loaded")

# LOAD LIAR DATASET
liar = pd.read_csv("dataset/train.tsv",sep="\t",header=None)

liar.columns = [
"id","label","statement","subject","speaker","speaker_job",
"state","party","barely_true","false","half_true","mostly_true",
"pants_fire","context"
]

label_map = {
"true":1,
"mostly-true":1,
"half-true":1,
"barely-true":0,
"false":0,
"pants-fire":0
}

liar["label"] = liar["label"].map(label_map)

liar = liar.rename(columns={"statement":"text"})

liar = liar[["text","label"]]

print("LIAR dataset loaded")

# LOAD FAKENEWSNET DATASET
polit_fake = pd.read_csv("dataset/PolitiFact_fake_news_content.csv")
polit_real = pd.read_csv("dataset/PolitiFact_real_news_content.csv")

polit_fake["label"] = 0
polit_real["label"] = 1

polit_fake["text"] = polit_fake["title"].astype(str) + " " + polit_fake["text"].astype(str)
polit_real["text"] = polit_real["title"].astype(str) + " " + polit_real["text"].astype(str)

polit_fake = polit_fake[["text","label"]]
polit_real = polit_real[["text","label"]]

fakenewsnet_data = pd.concat([polit_fake,polit_real])

print("FakeNewsNet dataset loaded")

# MERGE DATASETS
data = pd.concat([isot_data,welfake,liar,fakenewsnet_data])

data = data.dropna()
data = data.sample(frac=1).reset_index(drop=True)

print("Total dataset size:",data.shape)
print(data["label"].value_counts())

data.to_csv("final_dataset.csv", index=False)

# CLEAN TEXT
data["text"] = data["text"].apply(clean_text)

# TRAIN TEST SPLIT
X = data["text"]
y = data["label"]

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

print("Training samples:",len(X_train))
print("Testing samples:",len(X_test))

# TF-IDF VECTORIZATION
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7,
    max_features=30000,
    ngram_range=(1,2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("Text vectorization completed")

# SVM + GRID SEARCH
params = {
    "C":[0.1,1,5]
}

grid = GridSearchCV(
    LinearSVC(),
    params,
    cv=3,
    n_jobs=-1
)

grid.fit(X_train_vec,y_train)

model = grid.best_estimator_

print("Best parameter:",grid.best_params_)

# EVALUATE MODEL
y_pred = model.predict(X_test_vec)

print("\nAccuracy:",accuracy_score(y_test,y_pred))

print("\nClassification Report:")
print(classification_report(y_test,y_pred))

# SAVE MODEL
pickle.dump(model,open("model/fake_news_model.pkl","wb"))
pickle.dump(vectorizer,open("model/vectorizer.pkl","wb"))

print("\nModel saved successfully.")