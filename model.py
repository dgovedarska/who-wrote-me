import stylometry
import nltk
import pickle
import os
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify import ClassifierI
from nltk.corpus import gutenberg
from statistics import mode, StatisticsError

from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

class VoteClassifier(ClassifierI):
    def __init__(self, classifiers):
        self._classifiers = classifiers
    
    def classify(self, text_features):
        votes = []
        for classifier in self._classifiers: 
            vote = classifier.classify(text_features)
            votes.append(vote)
        try:
            m = mode(votes)
            return m
        except StatisticsError:
            return votes[0]

class Models:
    models = {}
    text_features = []

    def __init__(self, text_features):
        self.text_features = text_features
        if os.path.exists("./models/"):
            print("Prediction models exist. Loading them...")
            self.load_models()
            print("Prediction models loaded.")
        else:
            print("Trained prediction models not found. Training them... (this might take a while)")
            self.models["Multinomial Nayve Bayes Classifier"] = SklearnClassifier(MultinomialNB())
            self.models["Logistic Regression Classifier"] = SklearnClassifier(LogisticRegression())
            self.models["K-Nearest-Neighbors Classifier"] = SklearnClassifier(KNeighborsClassifier())
            self.models["Linear SVC Classifier"] = SklearnClassifier(LinearSVC())

            self.train_models()
            self.save_models()
            print("Training done.")
    
    def predict(self, text_features, algorythm="Vote Classifier"):
        return self.models[algorythm].classify(text_features)
   
    def train_models(self):
        voters = []
        for classifier in self.models:
            self.models[classifier].train(self.text_features)
            voters.append(self.models[classifier])
        """
        Vote Classifier is constructed from already trained classifiers!
        """
        self.models["Vote Classifier"] = VoteClassifier(voters)
    
    def load_models(self):
        self.load_model("Multinomial Nayve Bayes Classifier", "./models/MNB_classifier.pickle")
        self.load_model("Linear SVC Classifier", "./models/LSVC_classifier.pickle")
        self.load_model("Logistic Regression Classifier", "./models/LR_classifier.pickle")
        self.load_model("K-Nearest-Neighbors Classifier", "./models/KNN_classifier.pickle")
        self.load_model("Vote Classifier", "./models/VOTE_classifier.pickle")

    def save_model(self, model_identifier, model_file_name):
        try:
            with open(model_file_name, "wb") as model_file:
                pickle.dump(self.models[model_identifier], model_file)
        except IOError:
            print("Could not save model: ", model_identifier)
    
    def load_model(self, model_identifier, model_file_name):
        try:
            with open(model_file_name, "rb") as model_file:
               self.models[model_identifier] = pickle.load(model_file)
        except IOError:
            print("Could not load model: ", model_identifier)

    def save_models(self):
        if not os.path.exists("./models"):
            os.makedirs("./models")
        self.save_model("Multinomial Nayve Bayes Classifier", "./models/MNB_classifier.pickle")
        self.save_model("Linear SVC Classifier", "./models/LSVC_classifier.pickle")
        self.save_model("Logistic Regression Classifier", "./models/LR_classifier.pickle")
        self.save_model("K-Nearest-Neighbors Classifier", "./models/KNN_classifier.pickle")
        self.save_model("Vote Classifier", "./models/VOTE_classifier.pickle")