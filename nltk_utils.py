import numpy as np
import nltk
from nltk.stem import PorterStemmer
import re

# Initialize stemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    # Clean the sentence first
    sentence = re.sub(r'[^\w\s]', '', sentence)
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    # Stem all words in the sentence
    sentence_words = [stem(word) for word in tokenized_sentence]
    
    # Create bag of words
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in sentence_words:
            bag[idx] = 1
    return bag