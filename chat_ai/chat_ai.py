from typing import Mapping, Sequence
import numpy as np
from nltk.stem import WordNetLemmatizer
import nltk
import string
import random

import copy

from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
import tensorflow as tf

class PatternMan:

    def tokenize(self, pattern: str):
        tokens = nltk.word_tokenize(pattern)
        return tokens


class PatternWithParams(PatternMan):

    def tokenize(self, pattern: str):
        tokens = nltk.word_tokenize(pattern)

        res = []

        for idx, w in enumerate(tokens):
            if w.startswith(":") and idx > 0:
                if not tokens[idx-1].endswith("\\") : continue

            res.append(w)
                
        return res

class IntentMemoryCell:

    def __init__(self, selectedClasses : Sequence[str], chosenAnswer : str, nextExpectIntents : Sequence[str] = None, ocuurence = 1):
        self.selectedClasses = selectedClasses
        self.chosenAnswer = chosenAnswer
        self.nextExpectIntents = nextExpectIntents
        self.ocuurence = ocuurence

class ResponseMan:

    def getAnswer(self, memories, params):
        pass

class RMRepeatLastAnswer(ResponseMan):

    def __init__(self, name: str = "repeatLastAnswer"):
        self.__name = name


    def __get_name(self):
        return self.__name

    name = property(__get_name)

    def getAnswer(self, memories, params):
        intentMemory = memories['intent-history']
        m = intentMemory[len(intentMemory)-1]

        if m is None : return (None, False)

        if m.selectedClasses[0] == self.__name:
            m.ocuurence += 1
            return (m.chosenAnswer, False)

        return (m.chosenAnswer, True)
class ChatMan:

    def __init__(self, memorySize=5, responseManagers : Mapping[str, ResponseMan] = None):
        self.__memorySize = memorySize
        self.reset()

        if responseManagers is None:
            rmrla = RMRepeatLastAnswer()
            self.__responseManagers = {
                "repeatLastAnswer" : rmrla
            }
        else:
            self.__responseManagers = responseManagers

        
    def reset(self):
        self.__memories = dict()
        self.__memories['intent-history'] = [None] * self.__memorySize

    def memorizeIntent(self, mem : IntentMemoryCell):
        intentMemory = self.__memories['intent-history']

        for i in range(1, self.__memorySize):
            intentMemory[i-1] = intentMemory[i]

        intentMemory[self.__memorySize - 1] = mem
        

    def lastIntentMemory(self):
        intentMemory = self.__memories['intent-history']
        if len(intentMemory) > 0 : return copy.copy(intentMemory[self.__memorySize - 1])

        return None

    def selectClasses(self, classes):
        lastIntentMem = self.lastIntentMemory()

        if lastIntentMem is None: return classes

        newClasses = [c for c in classes if c in lastIntentMem.selectedClasses]

        return  newClasses if len(newClasses) > 0 else classes


    def chooseAnswer(self, jsonData, selectedTags):
        if jsonData is None: return None
        if 'intents' not in jsonData : return None

        tag = selectedTags[0]

        if tag not in jsonData['intents'] : return None

        if 'responses' not in jsonData['intents'][tag] : return None

        responses = jsonData['intents'][tag]['responses']

        rawAnswer = random.choice(responses)

        if rawAnswer is not None and rawAnswer.startswith("${") and rawAnswer.startswith("}"):
            p = rawAnswer.find(":", 2)
            command = rawAnswer[ 2 : p -1 ] if p > 0 else rawAnswer[ 2 : len(rawAnswer) -1 ]

            if p > 0 : params = rawAnswer[p+1:len(rawAnswer) -1]

            register = True
            if command in self.__responseManagers:
                rm = self.__responseManagers[command]
                newAnswer, register = rm.getAnswer(self.__memories, params)

                asw = newAnswer
            else:
                asw = rawAnswer

        if register:
            selectedIntentToMemorize = IntentMemoryCell(selectedTags, asw)
            self.memorizeIntent(selectedIntentToMemorize)

        return asw

    def __get_memories(self):
        return self.__memories

    memories = property(__get_memories)


class Chatbot:

    def __init__(self, jsonData, vocab, classes, patternManagers, fittedModel, defauthThreshold=0.5, lemmatizer = WordNetLemmatizer(), memorySize=5):
        self.__jsonData = jsonData
        self.__fittedModel = fittedModel
        self.__patternManagers = patternManagers
        self.__vocab = vocab
        self.__classes = classes
        self.__lemmatizer = lemmatizer
        self.__defauthThreshold = defauthThreshold
        
    def __bow(self, text):
        tokens = nltk.word_tokenize(text)
        tokens = [self.__lemmatizer.lemmatize(w.lower()) for w in tokens]
        
        bow = [0] * len(self.__vocab)
    
        for w in tokens:
            for idx, word in enumerate(self.__vocab):
                if word == w: bow[idx] = 1

        return np.array(bow)


    def answer(self, chatMan : ChatMan, text : str, threshold = None, noAnswerClass = "noAnswer"):
        if threshold is None: threshold = self.__defauthThreshold

        bow = self.__bow(text)

        result = self.__fittedModel.predict(np.array([bow]))[0]

        y_pred = [[idx, res] for idx, res in enumerate(result) if res > threshold]

        y_pred.sort(key=lambda x:x[1], reverse=True)

        selectedClasses = [self.__classes[r[0]] for r in y_pred]

        selectedClasses = chatMan.selectClasses(selectedClasses)

        if selectedClasses is None or len(selectedClasses) == 0:
            asw = chatMan.chooseAnswer(self.__jsonData, [noAnswerClass])
        else:
            asw = chatMan.chooseAnswer(self.__jsonData, selectedClasses)
        
        #selectedIntentToMemorize = IntentMemoryCell(selectedClasses, asw)

        #chatMan.memorizeIntent(selectedIntentToMemorize)
            
        return asw




class ChatbotTrainer:

    __defaultPatternMan = PatternMan()

    def __init__(self, patternManagers : Mapping[str, PatternMan]=None):
        self.__lemmatizer = WordNetLemmatizer()
        if patternManagers is None:
            self.__patternManagers = dict()
            self.__patternManagers["_default"] = ChatbotTrainer.__defaultPatternMan
        else:
            self.__patternManagers = patternManagers


    def addPatternMan(self, name : str, pm : PatternMan):
        self.__patternManagers[name] = pm

    def initialize():
        nltk.download("punkt")
        nltk.download("wordnet")
        nltk.download('omw-1.4')

    def __estimator(self, inputShape, outputShape):
        model = Sequential()
        model.add(Dense(128, input_shape=inputShape, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.3))
        model.add(Dense(outputShape, activation="softmax"))

        adam = tf.keras.optimizers.Adam(learning_rate=0.01, decay=1e-6)
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

        return model

    def train(self, jsonData,  epochs=200, threshold=0.5):

        vocab = []

        classes = []
        doc_X = []
        doc_Y = []
        
        intents = jsonData['intents']
        for tag in intents:
            intent = intents[tag]
            patterns = intent['patterns']
            
            for pattern in patterns:
                if isinstance(pattern, str) :
                    pm = ChatbotTrainer.__defaultPatternMan
                    tokens = pm.tokenize(pattern)
                    vocab.extend(tokens)
                    doc_X.append(pattern)
                    doc_Y.append(tag)

                else:
                    pm = self.__patternManagers[pattern['man']]
                    newPattern = pattern['pattern']

                    if isinstance(newPattern, str):
                        tokens = pm.tokenize(newPattern)
                        vocab.extend(tokens)
                        doc_X.append(newPattern)
                        doc_Y.append(tag)
                    else:
                        for p in newPattern:
                            tokens = pm.tokenize(p)
                            vocab.extend(tokens)
                            doc_X.append(p)
                            doc_Y.append(tag)
                
            if tag not in classes:
                classes.append(tag)
        
        
        vocab = [self.__lemmatizer.lemmatize(w.lower()) for w in vocab if w not in string.punctuation]

        vocab = sorted(set(vocab))
        classes = sorted(set(classes))

        training = []
        outEmpty = [0] * len(classes)

        for idx, doc in enumerate(doc_X):
            bow = []
            
            text = self.__lemmatizer.lemmatize(doc.lower())

            for w in vocab:
                bow.append(1) if w in text else bow.append(0)

            outputRow = list(outEmpty)
            outputRow[classes.index(doc_Y[idx])] = 1

            training.append([bow, outputRow])
        

        random.shuffle(training)
        training = np.array(training, dtype=object)

        train_X = np.array(list(training[:, 0]))

        train_Y = np.array(list(training[:, 1]))

        inputShape = (len(train_X[0]),)
        outputShape = (len(train_Y[0]))

        model = self.__estimator(inputShape, outputShape)

        model.fit(x=train_X, y=train_Y, epochs=epochs)

        #self.vocab = vocab
        #self.classes = classes
        #self.model = model

        return Chatbot(jsonData, vocab, classes, self.__patternManagers,  model, threshold)


