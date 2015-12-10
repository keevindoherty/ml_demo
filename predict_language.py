#!/usr/bin/env python
try:
	import cPickle as pickle
except:
	import pickle
import sys

__DOCS__ = """Type text and the classifier will guess if it is English, French, or Spanish \nTo exit type __EXIT__\nThe first time may take a second to load the classifier"""
classifier = None
def predict_language(text):
	global classifier
	if classifier is None:
		with open("svm.pkl", "rb") as f:
			classifier = pickle.load(f)
	prediction = classifier.predict([text])
	if prediction == 0:
		return 'English'
	elif prediction == 1:
		return 'French'
	elif prediction == 2:
		return "Spanish"
	else:
		print prediction
		return 'broken'


if __name__ == "__main__":
	print __DOCS__
	query = raw_input("Enter text: ")
	while query != "__EXIT__":
		print query
		print predict_language(query)
		query = raw_input("Enter text: ")