import nltk
import pickle
import os
import re
from os import listdir
from os.path import isfile, join
from nltk.tokenize import word_tokenize, sent_tokenize, PunktSentenceTokenizer
from nltk.corpus import gutenberg, stopwords
from nltk import FreqDist
from nltk.classify.scikitlearn import SklearnClassifier

# Generates lexicographical statistics for a text
# TODO think what to do about hardcoded paths
# TODO add exception handling here

BOOKS_DIR = './text_features_library/books'

class Text_Analysis:
    text_features_library = {}

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
            book_files = [f for f in listdir(BOOKS_DIR) if isfile(join(BOOKS_DIR, f))]
            labeled_books = ([(fileid, self.parse_author_name(fileid.split('-')[0])) for fileid in book_files])
            self.text_features_library = [(self.text_features(fileid), author) for (fileid, author) in labeled_books]

            self.save_text_features()
            print("Text features library generated.")
    
    def text_features(self, text):
        #TODO This might be stupid
        text_file = open(text, "rb")
        text = str(text_file.read())
        text_file.close()
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
        textWords = word_tokenize(text)
        unusual_words = self.find_unusual_words(textWords)
        content = [w for w in word_tokenize(text) if w.lower() in unusual_words]
        return round(len(content)/len(textWords)*100)

    def most_common_word_length(self, text):
        filtered_text = self.filter_stop_words(text)
        word_lengths = [len(w) for w in filtered_text]
        frequencyDistributionLenghts = FreqDist(len(word) for word in word_tokenize(text))
        return frequencyDistributionLenghts.max() # most common word length

    def average_sentence_length(self, text):
        num_sents = len(sent_tokenize(text))
        num_words = len(word_tokenize(text))
        return round(num_words/num_sents)

    def lexical_diversity(self, text):
        num_words = len(word_tokenize(text))
        num_vocab = len(set(w.lower() for w in word_tokenize(text)))
        return round(num_words/num_vocab)

    def add_author(self, author, texts):
        for text in texts:
            self.text_features_library.append((self.text_features(text), author))
        self.save_text_features()

    def get_authors(self):
        authors = []
        for entry in self.text_features_library:
            authors.append(entry[1])
        return set(authors)

    def parse_author_name(self, raw_name):
        author = [name for name in re.split(r'([A-Z][a-z]*)', raw_name) if name]
        return ' '.join(author)
      
    def save_text_features(self):
        if not os.path.exists("./text_features_library"):
            os.makedirs("./text_features_library")
        features_file = open("./text_features_library/features.pickle", "wb")
        pickle.dump(self.text_features_library, features_file)
        features_file.close()


#This works ok so far :)))

