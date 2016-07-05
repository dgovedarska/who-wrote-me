import stylometry
import nltk
from nltk.classify.scikitlearn import SklearnClassifier

from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# TODO Make this into a class

ta = stylometry.Text_Analysis()

#Not nice
train_set = ta.text_features_library[0:][::2]
test_set = ta.text_features_library[1:][::2]

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(train_set)
print("Multinomial Naive Bayes Algorithm accuracy percent:", (nltk.classify.accuracy(MNB_classifier, test_set))*100)

KNN_classifier = SklearnClassifier(KNeighborsClassifier())
KNN_classifier.train(train_set)
print("K Nearest Neighbors Algorithm accuracy percent:", (nltk.classify.accuracy(KNN_classifier, test_set))*100)

LR_classifier = SklearnClassifier(LogisticRegression())
LR_classifier.train(train_set)
print("Logistic Regression Algorithm accuracy percent:", (nltk.classify.accuracy(LR_classifier, test_set))*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(train_set)
print("Linear SVC Algorithm accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, test_set))*100)


