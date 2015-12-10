# Language Classification Demo
Some language classification with Python and scikit-learn

## Running the Demo
To try it out, unzip svm.zip in the repo directory, then run `predict_language.py`. It will prompt you to input some text, then attempt to guess the language of the input text.

## Ok but I want to do some machine learning
You can play around with the training code by first installing the dependencies:
* NumPy
* Pandas
* scikit-learn
* seaborn

On Linux you can run:

`sudo apt-get install python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose`

to get most of them, and use pip to get scikit-learn:

`pip install -U scikit-learn`

and seaborn:

`pip install seaborn`

You'll want to change the directories being used in the code and you'll need to get some data to learn from, I used Wikipedia dumps, but you can use anything you want. The parsers I wrote were specific to the documents I was using, however, so you'll need parsers specific to your content (or not if your documents are already separated).
