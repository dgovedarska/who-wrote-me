import unittest
import accuracy
import stylometry
import model
import ui
from unittest.mock import patch

text_analysis = stylometry.Text_Analysis()
models = model.Models(text_analysis.text_features_library)
accuracy = accuracy.accuracy()

class StylometryTest(unittest.TestCase):

    def test_filter_stop_words(self):
        r = text_analysis.filter_stop_words("I am the Cat!")
        self.assertEqual(r, ['I', 'Cat', '!'])

    def test_find_unusual_words(self):
        r = text_analysis.find_unusual_words(['buhta', 'kotka', 'cat'])
        self.assertEqual(r, ['buhta', 'kotka'])

    def test_unusual_words_content_fraction(self):
        r = text_analysis.unusual_words_content_fraction("kotka buhta cat dog")
        self.assertEqual(r, 50)

    def test_find_unusual_words(self):
        r = text_analysis.find_unusual_words(['buhta', 'kotka', 'cat'])
        self.assertEqual(r, ['buhta', 'kotka'])

    def test_most_common_word_length(self):
        r = text_analysis.most_common_word_length("kotka buhta cat dog egg")
        self.assertEqual(r, 3)

    def test_most_common_word_length_with_stopwords(self):
        r = text_analysis.most_common_word_length("kotka buhta cat dog egg am am am am")
        self.assertEqual(r, 3)

    def test_average_sentence_length(self):
        r = text_analysis.average_sentence_length("Hop trop. Hop. Trop trop. Hop hop.")
        self.assertEqual(r, 3)

    def test_lexical_diversity(self):
        r = text_analysis.lexical_diversity("hop hop hop trop trop trop")
        self.assertEqual(r, 3)

    def test_get_authors(self):
        r = text_analysis
        r.text_features_library = [({"feature": 4}, "Isaac Asimov"), ({"feature": 5}, "Leo Tolstoy"), ({"feature": 7}, "Leo Tolstoy")]
        self.assertEqual(r.get_authors(), {"Isaac Asimov", "Leo Tolstoy"})

    def test_parse_author_name(self):
        r = text_analysis.parse_author_name("JaneAusten")
        self.assertEqual(r, "Jane Austen")

    def test_text_features(self):
        r = text_analysis.text_features("./text.txt")
        self.assertEqual(r, {'lexical diversity': 1, 'average sentence length': 10, 'most common word length': 3, 'unusual words content fraction': 10})

# This fucks up the lib
'''
    def test_add_author(self):
        r = text_analysis
        r.text_features_library = [({"feature": 4}, "Isaac Asimov")]
        r.add_author("Me Me", ["ab ab ab"])
        self.assertIn("Me Me", r.get_authors())
'''
        # Perhaps a unit test for saving and loading the files?

class ModelTest(unittest.TestCase):
    def test_predict(self):
        text_analysis = stylometry.Text_Analysis()
        models = model.Models(text_analysis.text_features_library)
        r = models.predict({'lexical diversity': 1, 'average sentence length': 10, 'most common word length': 3, 'unusual words content fraction': 10},
        "Logistic Regression Classifier")
        self.assertIsInstance(r, str)

        r = models.predict({'lexical diversity': 1, 'average sentence length': 10, 'most common word length': 3, 'unusual words content fraction': 10},
        "K-Nearest-Neighbors Classifier")
        self.assertIsInstance(r, str)

        r = models.predict({'lexical diversity': 1, 'average sentence length': 10, 'most common word length': 3, 'unusual words content fraction': 10},
        "Linear SVC Classifier")
        self.assertIsInstance(r, str)

        r = models.predict({'lexical diversity': 1, 'average sentence length': 10, 'most common word length': 3, 'unusual words content fraction': 10},
        "Multinomial Nayve Bayes Classifier")
        self.assertIsInstance(r, str)

        r = models.predict({'lexical diversity': 1, 'average sentence length': 10, 'most common word length': 3, 'unusual words content fraction': 10},
        "Vote Classifier")
        self.assertIsInstance(r, str)



if __name__ == "__main__":
    unittest.main()