import nltk
import pickle
import os
from nltk.tokenize import word_tokenize, sent_tokenize, PunktSentenceTokenizer
from nltk.corpus import gutenberg, stopwords
from nltk import FreqDist
from nltk.classify.scikitlearn import SklearnClassifier

# Generates lexicographical statistics for a text
# TODO make sure the functions take list of words for text, not names
# TODO think what to do about hardcoded paths
# TODO add exception handling here
# TODO format author names properly
# TODO Adding Authors to library - wish to add more? - test that
# TODO Printing out all authors in the lib - This is what algos have trained on

class Text_Analysis:
    text_features_library = {}
    authors = () #better a set

    def __init__(self):
        if os.path.exists("./text_features_library/features.pickle"):
            print("Text features library exists. Loading it...")
            features_file = open("./text_features_library/features.pickle", "rb")
            self.text_features_library = pickle.load(features_file)
            features_file.close()
            print("Text features library loaded.")
        else:
            # Generating text features in a separate func
            print("Text features library does not exist. Generating it... (this may take a while)")
            labeled_books = ([(fileid, fileid.split('-')[0]) for fileid in gutenberg.fileids()])
            self.text_features_library = [(self.text_features(fileid), author) for (fileid, author) in labeled_books]

            save_text_features()
            print("Text features library generated.")
    
    def text_features(self, text):
        #filtered_text = filter_stop_words(text)
        return {'lexical diversity': self.lexical_diversity(text),
                'average sentence length': self.average_sentence_length(text),
                'most common word length': self.most_common_word_length(text),
                'unusual words content fraction': self.unusual_words_content_fraction(text)}  

    # Various lexicographical analysis functions
    def filter_stop_words(self, text):
        stop_words = set(stopwords.words("english"))
        text_words = word_tokenize(text)
        filtered_text = [w for w in text_words if w not in stop_words]
        return filtered_text

    def find_unusual_words(self, text):
        text_vocab = set(w.lower() for w in text if w.isalpha())
        english_vocab = set(w.lower() for w in nltk.corpus.words.words())
        unusual = text_vocab - english_vocab
        return sorted(unusual)

    def unusual_words_content_fraction(self, text):
        textWords = gutenberg.words(text)
        unusual_words = self.find_unusual_words(textWords)
        content = [w for w in gutenberg.words(text) if w.lower() in unusual_words]
        return round(len(content)/len(textWords)*100)

    def most_common_word_length(self, text):
        filtered_text = self.filter_stop_words(text)
        word_lengths = [len(w) for w in filtered_text]
        frequencyDistributionLenghts = FreqDist(len(word) for word in gutenberg.words(text))
        return frequencyDistributionLenghts.max() # most common word length

    def average_sentence_length(self, text):
        num_sents = len(gutenberg.sents(text))
        num_words = len(gutenberg.words(text))
        return round(num_words/num_sents)

    def lexical_diversity(self, text):
        num_words = len(gutenberg.words(text))
        num_vocab = len(set(w.lower() for w in gutenberg.words(text)))
        return round(num_words/num_vocab)

    def add_author(self, author, texts):
        for text in texts:
            self.text_features_library.append((self.text_features(text), author))
        save_text_features()
        
    def save_text_features(self):
        if not os.path.exists("./text_features_library"):
            os.makedirs("./text_features_library")
        features_file = open("./text_features_library/features.pickle", "wb")
        pickle.dump(self.text_features_library, features_file)
        features_file.close()


#This works ok so far :)))

#ta = Text_Analysis()
#print(ta.text_features_library)
