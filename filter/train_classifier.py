#!/usr/bin/env python
import sys
try:
	import cPickle as pickle
except:
	import pickle
import numpy as np
import pandas as pd
from utils import parsers
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

DATA_DIR = '~/dotGit/ml_demo/docs/'
names = ['doc','label']
label_names = ['en', 'fr', 'es']

def load_data():
	# Load all data into memory 
	print "Loading data"

	# Load data, parse using the parsers, re-encode as utf-8, and label
	en_data = parsers.parse_en(open(DATA_DIR+'english/en_corpus.txt', 'r').read())
	en_df = pd.DataFrame({'doc': pd.Series([doc.decode('latin-1').encode('utf-8') for doc in en_data][:100]), 'label': 0})

	fr_data = parsers.parse_fr(open(DATA_DIR+'french/fr_corpus.txt', 'r').read())
	fr_df = pd.DataFrame({'doc': pd.Series([doc.decode('latin-1').encode('utf-8') for doc in fr_data][:100]), 'label': 1})
	
	es_data = parsers.parse_es(open(DATA_DIR+'spanish/es_corpus.txt', 'r').read())
	es_df = pd.DataFrame({'doc': pd.Series([doc.decode('latin-1').encode('utf-8') for doc in es_data][:100]), 'label': 2})


	# Append the dataframes
	data = pd.concat([en_df, fr_df, es_df])
	print "Data loaded successfully"
	return data

def save_classifier(classifier):
	# Pickle the classifier so we can use it later
	with open("svm.pkl", "wb") as f:
		pickle.dump(classifier, f)
	print "Classifier saved to svm.pkl"

def train_classifier(data, train_percent):
	# Use sklearn split function to randomize and split training and test data
	train, test = train_test_split(data, train_size=train_percent)
	train_data, test_data = pd.DataFrame(train, columns=names), pd.DataFrame(test, columns=names)

	# Set up an sklearn pipeline for our data
	#	- uses Support Vector Machine classifier
	classifier = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1, 6),
                             analyzer='char')),
					 	('clf', SGDClassifier(loss='hinge', penalty='l2',
						 	alpha=1e-3, n_iter=5))
	])

	# Fit a classifier to the training data
	classifier = classifier.fit(train_data['doc'], train_data['label'])

	print "SVM trained successfully, testing now..."

	# Predict the labels for the test data to obtain performance metrics
	y_predicted = classifier.predict(test_data['doc'])

	print "Tests successful, calculating metrics..."
	# Obtain metrics for accuracy and recall
	recall = metrics.recall_score(test_data['label'], y_predicted)
	cm = metrics.confusion_matrix(test_data['label'], y_predicted)
	plt.figure(figsize=(8, 8))
	sns.heatmap(cm, annot=True,  fmt='', xticklabels=label_names, yticklabels=label_names,cmap="YlGnBu");
	plt.title('Confusion Matrix for Languages');
	plt.show()
	print "Accuracy = " + str(np.mean(y_predicted == test_data['label']))
	print "Recall = " + str(recall)
	return classifier

def train_tuned_classifier(data):
	# Create a dataframe of the training data (all of it)
	train = data
	train_data = pd.DataFrame(train, columns=names)

	# Set up an sklearn pipeline for our data
	#	- uses Support Vector Machine classifier
	classifier = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1, 6),
                             analyzer='char')),
					 	('clf', SGDClassifier(loss='hinge', penalty='l2',
						 	alpha=1e-3, n_iter=5))
	])

	# Fit a classifier to the training data
	classifier = classifier.fit(train_data['doc'], train_data['label'])

	print "SVM trained successfully."
	return classifier

if __name__ == "__main__":
	data = load_data()
	if len(sys.argv) == 2 and sys.argv[1] == '--validation':
		classifier = train_classifier(data, 0.7)
	else:
		classifier = train_tuned_classifier(data)
		save_classifier(classifier)