import lexicography
import nltk
import pickle
import random
from nltk.tokenize import word_tokenize, sent_tokenize, PunktSentenceTokenizer
from nltk.corpus import gutenberg
from nltk import FreqDist
from nltk.classify.scikitlearn import SklearnClassifier

from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier


def unusualWords(text):
    text_vocab = set(w.lower() for w in text if w.isalpha())
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())
    unusual = text_vocab - english_vocab
    return sorted(unusual)

def unusualWordsContentFraction(text):
     textWords = gutenberg.words(text)
     unusualwords = unusualWords(textWords)
     content = [w for w in gutenberg.words(text) if w.lower() in unusualwords]
     return round(len(content)/len(textWords)*100)

def mostCommonWordLength(text): 
    wordLengths = [len(w) for w in gutenberg.words(text)]
    frequencyDistributionLenghts = FreqDist(len(word) for word in gutenberg.words(text))
    return frequencyDistributionLenghts.max() # most common word length

def averageSentenceLength(text):
    num_sents = len(gutenberg.sents(text))
    num_words = len(gutenberg.words(text))
    return round(num_words/num_sents)

def lexicalDiversity(text):
    num_words = len(gutenberg.words(text))
    num_vocab = len(set(w.lower() for w in gutenberg.words(text)))
    return round(num_words/num_vocab)

def textFeatures(text):
    return {'lexicalDiversity': lexicalDiversity(text),
    'averageSentenceLength': averageSentenceLength(text),
    'mostCommonWordLength': mostCommonWordLength(text),
    'unusualWordsContentFraction': unusualWordsContentFraction(text)}


labeled_books = ([(fileid, fileid.split('-')[0]) for fileid in gutenberg.fileids()])

#featuresets = [(textFeatures(fileid), author) for (fileid, author) in labeled_books]


#featureSetsFile = open("features.pickle", "wb")
#pickle.dump(featuresets, featureSetsFile)
#featureSetsFile.close()

featuresets = pickle.load(open( "features.pickle", "rb"))
random.shuffle(featuresets)
train_set, test_set = featuresets[:50], featuresets[50:] 

print(featuresets)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(train_set)

'''
MNBClassifierFile = open("MNB_classifier.pickle", "rb")
MNB_classifier = pickle.load(MNBClassifierFile)
MNBClassifierFile.close()

saveMNBClassifier = open("MNB_classifier.pickle", "wb")
pickle.dump(MNB_classifier, saveMNBClassifier)
saveMNBClassifier.close()
'''
print("Multinomial Naive Bayes Algorithm accuracy percent:", (nltk.classify.accuracy(MNB_classifier, test_set))*100)

KNN_classifier = SklearnClassifier(KNeighborsClassifier())
KNN_classifier.train(train_set)
print("K Nearest Neighbors Algorithm accuracy percent:", (nltk.classify.accuracy(KNN_classifier, test_set))*100)


