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

BOOKS_DIR = './text_features_library/books'
FEATURES_FILE = './text_features_library/features.pickle'

class Text_Analysis:
    text_features_library = {}

    def __init__(self):
        if os.path.exists(FEATURES_FILE):
            print("Text features library exists. Loading it...")
            try:
                with open (FEATURES_FILE, "rb") as features_file:
                    self.text_features_library = pickle.load(features_file)
                    print("Text features library loaded.")
            except IOError:
                print("Could not load text features library.")
        else:
            print("Text features library does not exist. Generating it... (this may take a while)")
            book_files = [f for f in listdir(BOOKS_DIR) if isfile(join(BOOKS_DIR, f))]
            labeled_books = ([(fileid, self.parse_author_name(fileid.split('-')[0])) for fileid in book_files])
            self.text_features_library = [(self.text_features(join(BOOKS_DIR,fileid)), author) for (fileid, author) in labeled_books]

            self.save_text_features()
            print("Text features library generated.")
    
    def text_features(self, text):
        try:
            with open(text, "rb") as text_file:
                text_content = str(text_file.read())
                return {'lexical diversity': self.lexical_diversity(text_content),
                        'average sentence length': self.average_sentence_length(text_content),
                        'most common word length': self.most_common_word_length(text_content),
                        'unusual words content fraction': self.unusual_words_content_fraction(text_content)}
        except IOError:
            print("Could not open file: ", text)     

    """
    Various text analysis functions:
    """
    def filter_stop_words(self, text):
        """
        Stop words are words which are used a lot, but have little meaning: am, the etc.
        """
        stop_words = set(stopwords.words("english"))
        text_words = word_tokenize(text)
        filtered_text = [w for w in text_words if w not in stop_words]
        return filtered_text

    def find_unusual_words(self, text):
        """
        Unusual words are words outside of the english vocabulary: kuche, kotka, etc.
        """
        text_vocab = set(w.lower() for w in text if w.isalpha())
        english_vocab = set(w.lower() for w in nltk.corpus.words.words())
        unusual = text_vocab - english_vocab
        return sorted(unusual)

    def unusual_words_content_fraction(self, text):
        text_words = word_tokenize(text)
        unusual_words = self.find_unusual_words(text_words)
        content = [w for w in word_tokenize(text) if w.lower() in unusual_words]
        if len(text_words) != 0:
            return round(len(content)/len(text_words)*100)
        else:
            return 0

    def most_common_word_length(self, text):
        filtered_text = self.filter_stop_words(text)
        if len(filtered_text) == 0:
            return 0
        if len(filtered_text) == 1:
            return len(filtered_text[0])
        else:
            frequencyDistributionLenghts = FreqDist(len(word) for word in filtered_text)
            return frequencyDistributionLenghts.max() # most common word length

    def average_sentence_length(self, text):
        num_sents = len(sent_tokenize(text))
        num_words = len(word_tokenize(text))
        if num_sents == 0:
            return num_words
        else:
            return round(num_words/num_sents)

    def lexical_diversity(self, text):
        num_words = len(word_tokenize(text))
        num_vocab = len(set(w.lower() for w in word_tokenize(text)))
        if num_vocab == 0:
            return 0
        else:
            return round(num_words/num_vocab)

    def add_author(self, author, texts):
        for text in texts:
            result = self.text_features(text)
            if result is None:
                print('Something went wrong!')
                return
            else:
                self.text_features_library.append((result, author))
        self.save_text_features()
        print('Author successfuly added!')

    def get_authors(self):
        authors = []
        for entry in self.text_features_library:
            authors.append(entry[1])
        return set(authors)

    def parse_author_name(self, raw_name):
        """
        We must parse the author names coming from the default library. They look like this:
        NameName
        """
        author = [name for name in re.split(r'([A-Z][a-z]*)', raw_name) if name]
        return ' '.join(author)
      
    def save_text_features(self):
        if not os.path.exists("./text_features_library"):
            os.makedirs("./text_features_library")
        try:
            with open("./text_features_library/features.pickle", "wb") as features_file:
                pickle.dump(self.text_features_library, features_file)
        except IOError:
            print("Could not save text features library!")

    def load_text_features(self):
        try:
            with open("./text_features_library/features.pickle", "rb") as features_file:
                self.text_features_library = pickle.load(features_file)
        except IOError:
            print("Could not load text features library!")
