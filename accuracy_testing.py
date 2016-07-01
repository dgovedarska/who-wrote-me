import lexicography
import nltk
import pickle
import random
from nltk.tokenize import word_tokenize, sent_tokenize, PunktSentenceTokenizer
from nltk.corpus import stopwords, state_union, gutenberg
from nltk.classify.scikitlearn import SklearnClassifier

from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier


def averageWordLength(text): # not too good since it's typical for english
    num_chars = len(gutenberg.raw(text))
    num_words = len(gutenberg.words(text))
    return round(num_chars/num_words)

def averageSentenceLength(text):
    num_sents = len(gutenberg.sents(text))
    num_words = len(gutenberg.words(text))
    return round(num_words/num_sents)

def lexicalDiversity(text):
    num_words = len(gutenberg.words(text))
    num_vocab = len(set(w.lower() for w in gutenberg.words(text)))
    return round(num_words/num_vocab)

def textFeatures(text):
    return {'ld': lexicalDiversity(text), 'asl': averageSentenceLength(text)}


labeled_books = ([(fileid, fileid.split('-')[0]) for fileid in gutenberg.fileids()])
random.shuffle(labeled_books)


featuresets = [(textFeatures(fileid), author) for (fileid, author) in labeled_books]
train_set, test_set = featuresets[:8], featuresets[8:]



MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(train_set)
print("Multinomial Naive Bayes Algorithm accuracy percent:", (nltk.classify.accuracy(MNB_classifier, test_set))*100)

KNN_classifier = SklearnClassifier(KNeighborsClassifier())
KNN_classifier.train(train_set)
print("K Nearest Neighbors Algorithm accuracy percent:", (nltk.classify.accuracy(KNN_classifier, test_set))*100)


