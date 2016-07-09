import stylometry
import nltk
from nltk.classify.scikitlearn import SklearnClassifier
import model

from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

class accuracy:
    _models = {}

    def __init__(self):
        print("Initializing accuracy tests environment...")
        self._text_analysis = stylometry.Text_Analysis()
        self.initialize_models()
        """
        Train set and test set are equal parts of the original text features set.
        Original set is sorted alphabetically and we want every author appearing
        in both sets, so we take the odd and even elements for the sets.
        """
        self._train_set = self._text_analysis.text_features_library[0:][::2]
        self._test_set = self._text_analysis.text_features_library[1:][::2]
        self.train_models()

    def initialize_models(self):
        self._models["Multinomial Nayve Bayes Classifier"] = SklearnClassifier(MultinomialNB())
        self._models["Logistic Regression Classifier"] = SklearnClassifier(LogisticRegression())
        self._models["K-Nearest-Neighbors Classifier"] = SklearnClassifier(KNeighborsClassifier())
        self._models["Linear SVC Classifier"] = SklearnClassifier(LinearSVC())
        
    def train_models(self):
        for classifier in self._models:
            self._models[classifier].train(self._train_set)
    
    def test_accuracy(self):
        for classifier_name in self._models:
            classifier = self._models[classifier_name]
            print(classifier_name + " Algorithm accuracy percent:", 
            (nltk.classify.accuracy(classifier, self._test_set))*100)
