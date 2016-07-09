import unittest
import accuracy
import stylometry
import model
import ui
from nltk.classify.scikitlearn import SklearnClassifier

text_analysis = stylometry.Text_Analysis()
models = model.Models(text_analysis.text_features_library)

class StylometryTest(unittest.TestCase):
    def test_filter_stop_words(self):
        r = text_analysis.filter_stop_words("I am the Cat!")
        self.assertEqual(r, ['I', 'Cat', '!'])

    def test_filter_stop_words_empty_list(self):
        r = text_analysis.filter_stop_words("")
        self.assertEqual(r, [])

    def test_find_unusual_words(self):
        r = text_analysis.find_unusual_words(['buhta', 'kotka', 'cat'])
        self.assertEqual(r, ['buhta', 'kotka'])

    def test_unusual_words_content_fraction(self):
        r = text_analysis.unusual_words_content_fraction("kotka buhta cat dog")
        self.assertEqual(r, 50)

    def test_unusual_words_content_fraction_all_unusual(self):
        r = text_analysis.unusual_words_content_fraction("kotka buhta")
        self.assertEqual(r, 100)

    def test_unusual_words_content_fraction_no_unusual(self):
        r = text_analysis.unusual_words_content_fraction("cat dog")
        self.assertEqual(r, 0)

    def test_unusual_words_content_fraction_no_words(self):
        r = text_analysis.unusual_words_content_fraction("")
        self.assertEqual(r, 0)

    def test_find_unusual_words(self):
        r = text_analysis.find_unusual_words(['buhta', 'kotka', 'cat'])
        self.assertEqual(r, ['buhta', 'kotka'])

    def test_find_unusual_words_no_unusual_words(self):
        r = text_analysis.find_unusual_words(['cat'])
        self.assertEqual(r, [])

    def test_find_unusual_words_no_words(self):
        r = text_analysis.find_unusual_words([])
        self.assertEqual(r, [])

    def test_most_common_word_length(self):
        r = text_analysis.most_common_word_length("kotka buhta cat dog egg")
        self.assertEqual(r, 3)

    def test_most_common_word_length_no_words(self):
        r = text_analysis.most_common_word_length("")
        self.assertEqual(r, 0)

    def test_most_common_word_length_one_word(self):
        r = text_analysis.most_common_word_length("cat")
        self.assertEqual(r, 3)

    def test_most_common_word_length_with_stopwords(self):
        r = text_analysis.most_common_word_length("kotka buhta cat dog egg am am am am")
        self.assertEqual(r, 3)

    def test_average_sentence_length(self):
        r = text_analysis.average_sentence_length("Hop trop. Hop. Trop trop. Hop hop.")
        self.assertEqual(r, 3)

    def test_average_sentence_length_no_words(self):
        r = text_analysis.average_sentence_length("")
        self.assertEqual(r, 0)

    def test_average_sentence_length_one_sent(self):
        r = text_analysis.average_sentence_length("one two one two")
        self.assertEqual(r, 4)

    def test_lexical_diversity(self):
        r = text_analysis.lexical_diversity("hop hop hop trop trop trop")
        self.assertEqual(r, 3)

    def test_lexical_diversity_no_words(self):
        r = text_analysis.lexical_diversity("")
        self.assertEqual(r, 0)

    def test_lexical_diversity_one_word(self):
        r = text_analysis.lexical_diversity("one")
        self.assertEqual(r, 1)

    def test_get_authors(self):
        r = text_analysis
        r.text_features_library = [({"feature": 4}, "Isaac Asimov"), ({"feature": 5}, "Leo Tolstoy"), ({"feature": 7}, "Leo Tolstoy")]
        self.assertEqual(r.get_authors(), {"Isaac Asimov", "Leo Tolstoy"})

    def test_get_authors_no_authors(self):
        r = text_analysis
        r.text_features_library = []
        self.assertEqual(r.get_authors(), set())

    def test_parse_author_name(self):
        r = text_analysis.parse_author_name("JaneAusten")
        self.assertEqual(r, "Jane Austen")
    
    def test_parse_author_name_empty(self):
        r = text_analysis.parse_author_name("")
        self.assertEqual(r, "")

    def test_text_features(self):
        r = text_analysis.text_features("./text.txt")
        self.assertEqual(r, {'lexical diversity': 1, 'average sentence length': 10,
        'most common word length': 3, 'unusual words content fraction': 10})

    def test_text_features_empty(self):
        r = text_analysis.text_features("./empty.txt")
        self.assertEqual(r, {'lexical diversity': 1, 'average sentence length': 2,
        'most common word length': 1, 'unusual words content fraction': 0})

    def test_add_author_fail(self):
        self.assertEquals(None, text_analysis.add_author("Test FAIL", ["ab ab ab"]))
        self.assertNotIn("Test FAIL", text_analysis.get_authors())

    def test_add_author_ok(self):
        self.assertEquals(None, text_analysis.add_author("Test OK", ['./text.txt']))
        self.assertIn("Test OK", text_analysis.get_authors())


class ModelTest(unittest.TestCase):
    def test_init(self):
        text_analysis = stylometry.Text_Analysis()
        r = model.Models(text_analysis.text_features_library)
        for model_name in r.models:
            if model_name != "Vote Classifier":
                self.assertIsInstance(r.models[model_name], SklearnClassifier)
            else:
                self.assertIsInstance(r.models[model_name], model.VoteClassifier)

    def test_predict(self):
        text_analysis = stylometry.Text_Analysis()
        models = model.Models(text_analysis.text_features_library)
        test_features = {'lexical diversity': 1, 'average sentence length': 10,
        'most common word length': 3, 'unusual words content fraction': 10}
        r = models.predict(test_features, "Logistic Regression Classifier")
        self.assertIsInstance(r, str)

        r = models.predict(test_features, "K-Nearest-Neighbors Classifier")
        self.assertIsInstance(r, str)

        r = models.predict(test_features, "Linear SVC Classifier")
        self.assertIsInstance(r, str)

        r = models.predict(test_features, "Multinomial Nayve Bayes Classifier")
        self.assertIsInstance(r, str)

        r = models.predict(test_features, "Vote Classifier")
        self.assertIsInstance(r, str)

        r = models.predict(test_features)
        self.assertIsInstance(r, str)

class AccuracyTest(unittest.TestCase):
    def test_init(self):
        r = accuracy.accuracy()
        for model_name in r._models:
            self.assertIsInstance(r._models[model_name], SklearnClassifier)
        self.assertIsInstance(r._text_analysis, stylometry.Text_Analysis)

if __name__ == "__main__":
    unittest.main()