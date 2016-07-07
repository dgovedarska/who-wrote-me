import stylometry
import nltk
import pickle
import os
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify import ClassifierI
from nltk.corpus import gutenberg
from statistics import mode

from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# TODO more algos
# TODO hardcoded paths
# TODO improve loading and saving
# TODO rename them to classifiers

#again class and if models are not yet trained - train them and tell the user - can be demonstrated because it's relatively fast

class vote_classifier(ClassifierI):
    def __init__(self, classifiers):
        self._classifiers = classifiers
    
    def classify(self, text_features):
        votes = []
        for classifier in self._classifiers:
            vote = classifier.classify(text_features)
            votes.append(vote)
        print(votes)
        print(mode(votes))
        return mode(votes)
    
    def confidence(self, features):
        votes = []
        for classifier in self._classifiers:
            votes.append(classifier.predict(text_features))
        
        choice_votes = votes.count(mode(votes))
        confidence_percent = choice_votes / len(votes)
        return confidence_percent

class Models:
    models = {} # that's a dict
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
        for classifier in self.models:
            self.models[classifier].train(self.text_features)
        self.models["Vote Classifier"] = SklearnClassifier(vote_classifier([self.models[name] for name in self.models]))
    
    def load_models(self):
    # Exception handling
    # Voting one as well
        MNB_classifier_file = open("./models/MNB_classifier.pickle", "rb")
        MNB_classifier = pickle.load(MNB_classifier_file)
        MNB_classifier_file.close()
        self.models["Multinomial Nayve Bayes Classifier"] = MNB_classifier

        LR_classifier_file = open("./models/LR_classifier.pickle", "rb")
        LR_classifier = pickle.load(LR_classifier_file)
        LR_classifier_file.close()
        self.models["Logistic Regression Classifier"] = LR_classifier

        KNN_classifier_file = open("./models/KNN_classifier.pickle", "rb")
        KNN_classifier = pickle.load(KNN_classifier_file)
        KNN_classifier_file.close()
        self.models["K-Nearest-Neighbors Classifier"] = KNN_classifier

        LSVC_classifier_file = open("./models/LSVC_classifier.pickle", "rb")
        LSVC_classifier = pickle.load(LSVC_classifier_file)
        LSVC_classifier_file.close()
        self.models["Linear SVC Classifier"] = LSVC_classifier

        VOTE_classifier_file = open("./models/VOTE_classifier.pickle", "rb")
        VOTE_classifier = pickle.load(VOTE_classifier_file)
        VOTE_classifier_file.close()
        self.models["Vote Classifier"] = LSVC_classifier

    def save_models(self):
        #Exception handling
        #Voting algorythm as well
        if not os.path.exists("./models"):
            os.makedirs("./models")
        MNB_classifier_file = open("./models/MNB_classifier.pickle", "wb")
        pickle.dump(self.models["Multinomial Nayve Bayes Classifier"], MNB_classifier_file)
        MNB_classifier_file.close()

        LSVC_classifier_file = open("./models/LSVC_classifier.pickle", "wb")
        pickle.dump(self.models["Linear SVC Classifier"], LSVC_classifier_file)
        LSVC_classifier_file.close()

        LR_classifier_file = open("./models/LR_classifier.pickle", "wb")
        pickle.dump(self.models["Logistic Regression Classifier"], LR_classifier_file)
        LR_classifier_file.close()

        KNN_classifier_file = open("./models/KNN_classifier.pickle", "wb")
        pickle.dump(self.models["K-Nearest-Neighbors Classifier"], KNN_classifier_file)
        KNN_classifier_file.close()

        VOTE_classifier_file = open("./models/VOTE_classifier.pickle", "wb")
        pickle.dump(self.models["Vote Classifier"], VOTE_classifier_file)
        VOTE_classifier_file.close()

        


'''
ta = stylometry.Text_Analysis()
#print(ta.text_features_library)

m = Models(ta.text_features_library)
m.save_models()
files = gutenberg.fileids()
print(files[20])
print(ta.text_features(files[20]))
print(m.predict(ta.text_features(files[20]), "Logistic Regression Classifier"))

models = []
for key in m.models:
    models.append(m.models[key])


vc = vote_classifier(models)
#print(vc._classifiers)

print(vc.classify(ta.text_features(files[20])))
'''