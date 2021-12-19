import numpy
import random
import json
import pickle

#lemmatisation => Prendre la racine du mot reconnu pour effetuer un traitement
#tokennize => Separe les mots pour les analyser séparément

import nltk
#nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

#Les intents sont lues depuis le fichier intents
intents = json.loads(open("intents.json").read())

words = []
classes = []
documents = []
ignoredLetters = ["?", "!", ",", "."]

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

print(documents)

