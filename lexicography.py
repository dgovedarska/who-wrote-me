import nltk
from nltk.tokenize import word_tokenize, sent_tokenize, PunktSentenceTokenizer
from nltk.corpus import stopwords, state_union, gutenberg

# PunktSentenceTokenizer is unsupervised machine learing sentence tokenizer. Pretrained and you can also train it again.

# Generates lexicographical statistics for a text
def getFeatures(text):
    filtered_text = filter_stop_words(text)
    # average words per sentence
    # average word length

    print("features for " + text)

  

# Various lexicographical analysis functions
def filter_stop_words(text):
    stop_words = set(stopwords.words("english"))
    text_words = word_tokenize(text)
    filtered_text = [w for w in text_words if w not in stop_words]
    return filtered_text

def part_of_speech_tagging(text):
    training_text = state_union.raw("2005-GWBush.txt")
    custom_sent_tokenizer = PunktSentenceTokenizer(training_text)
    tokenized_text = custom_sent_tokenizer.tokenize(text)
    tagged = []
    try:
        for i in tokenized_text:
            words = nltk.word_tokenize(i)
            tagged.append(nltk.pos_tag(words))

        return tagged
    except Exception as e:
        print(str(e))


def argWordsPerSentence(text): pass

def argCommasPerSentence(text): pass

def argColonsPerSentence(text): pass

def argSemicolonsPerSentence(text): pass

def lexicalDiversity(text): pass

def commonWords(text): pass

# here we'll test the functions


#print(sent_tokenize(example_text))
#print(word_tokenize(example_text))




# ...etc. I'll try to implement as many as possible
