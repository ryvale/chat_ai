#from tokenize import Number
from typing import Iterable, Mapping, Callable, Sequence
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
import nltk
import string
import random

import copy

from keras.models import Sequential
from keras.layers import Dense, Dropout
#from tensorflow.keras.optimizers import SGD
import tensorflow as tf

import spacy



class WordMan:

    def __init__(self, stemmer, lematizer):
        self.__stemmer = stemmer
        self.__lematizer = lematizer

    def __get_stemmer(self):
        return self.__stemmer

    stemmer = property(__get_stemmer)

    def __get_lematizer(self):
        return self.__lematizer

    lematizer = property(__get_lematizer)


    def stem(self, sent):
        return self.__stemmer.stem(sent)

    def lemmatize(self, sent):
        return self.__lematizer.lemmatize(sent)


    def tokenize(self, sent, transform = lambda s : s):
        pass

class RawSpacyWordMan(WordMan):

    def __init__(self, langRef : str, stemmLang : str, lematizer):
        super().__init__(SnowballStemmer(language=stemmLang), lematizer)
        self.__nlp = spacy.load(langRef)

    def rawTokenize(self, sent):
        doc = self.__nlp(sent)
        
        return [t.text for t in doc]

    def tokenize(self, sent, transform = lambda s : s):
        if transform is None: return self.rawTokenize(sent)

        doc = self.__nlp(sent)
        
        return [transform(t.text) for t in doc], [t.text for t in doc]

class DefaultWordMan(RawSpacyWordMan):

    def __init__(self, langRef : str, stemmLang : str, lematizer = WordNetLemmatizer()):
        super().__init__(langRef, stemmLang, lematizer)

    def __bringCloser(t : str, symbs : Iterable[str] = ["'"]):
        t1 = t
        for s in symbs:
            t2 = t1.replace(s + ' ', s)
            t3 = t1.replace(' ' + s, s)
            
            while t2 != t1 or t3 != t1:
                t1 = t2.replace(' ' + s, s)
                
                t2 = t1.replace(s + ' ', s)
                t3 = t1.replace(' ' + s, s)
                
        return t1

    def tokenize(self, sent, transform = lambda s : s.lower()):
        doc = DefaultWordMan.__bringCloser(sent)

        tokens = self.rawTokenize(doc)

        if transform is None: return tokens, tokens

        return [transform(w) for w in tokens], tokens

class DefaultWordManWithParams(DefaultWordMan):

    def __init__(self, langRef : str, stemmLang : str, lematizer = WordNetLemmatizer()):
        super().__init__(langRef, stemmLang, lematizer)

    def tokenize(self, sent, transform = lambda s : s.lower()):
        tres = super().tokenize(sent, transform)

        tokens = tres[0]

        res = []

        for idx, w in enumerate(tokens):
            if w.startswith(":") and idx > 0:
                if not tokens[idx-1].endswith("\\") : continue

            res.append(w)
                
        return res, tres[1]

class PatternMan:

    def tokenize(self, pattern: str):
        tokens = nltk.word_tokenize(pattern)
        return tokens

    def extractParams(message : str) -> Mapping[str, str]:
        return None


class PatternWith1Params(PatternMan):

    def tokenize(self, pattern: str):
        tokens = nltk.word_tokenize(pattern)

        res = []

        for idx, w in enumerate(tokens):
            if w.startswith(":") and idx > 0:
                if not tokens[idx-1].endswith("\\") : continue

            res.append(w)
                
        return res

    def __bowExtraction(self, messageTokens,  pattern):
        patternTokens = nltk.word_tokenize(pattern.lower())

        return [ w for w in messageTokens if w.lower() not in patternTokens ]


    def extractParams(self, message : str, patterns : Iterable[object]) -> Mapping[str, str]:
        messageTokens = nltk.word_tokenize(message)

        paramsNames = []

        for idx, w in enumerate(messageTokens):
            if w.startswith(":"):
                if idx == 0: paramsNames.append(w)

                if messageTokens[idx - 1].endswith("\\") : continue

                paramsNames.append(w)
        
        nbParams = len(paramsNames)
        if nbParams == 0: return None

        values = []
        for pattern in patterns:
            values = values + self.__bowExtraction(messageTokens, pattern)

        countDict = dict()
        for v in values:
            if v in countDict:
                countDict[v] += 1
            else:
                countDict[v] = 1

        maxCount = 0
        res = None
        for v in countDict:
            if countDict[v] > maxCount:
                maxCount = countDict[v]
                res = v

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


    def chooseAnswer(self, jsonData, text:str, selectedTags : Sequence[str]):
        if jsonData is None: return None
        if 'intents' not in jsonData : return None

        tag = selectedTags[0]

        if tag not in jsonData['intents'] : return None

        if 'responses' not in jsonData['intents'][tag] : return None

        tagConfig = jsonData['intents'][tag]

        responses = tagConfig['responses']

        rawAnswer = random.choice(responses)

        register = True
        asw = rawAnswer
        if rawAnswer is not None and rawAnswer.startswith("${") and rawAnswer.endswith("}"):
            p = rawAnswer.find(":", 2)
            command = rawAnswer[ 2 : p -1 ] if p > 0 else rawAnswer[ 2 : len(rawAnswer) -1 ]

            params = None
            if p > 0 : params = rawAnswer[p+1:len(rawAnswer) -1]

            if command in self.__responseManagers:
                rm = self.__responseManagers[command]
                newAnswer, register = rm.getAnswer(self.__memories, params)

                asw = newAnswer
        
        if register:
            selectedIntentToMemorize = IntentMemoryCell(selectedTags, asw)
            self.memorizeIntent(selectedIntentToMemorize)

        return asw

    def __get_memories(self):
        return self.__memories

    memories = property(__get_memories)


class Chatbot:

    def __init__(self, jsonData, vocab, classes, wordMans, defaultWordMan, addFeatures : Callable[[str, Sequence[str]], Sequence[object]], fittedModel, defauthThreshold=0.5, memorySize=5, transformToken : Callable[[str] , str] = None):
        self.__jsonData = jsonData
        self.__fittedModel = fittedModel
        self.__wordMans = wordMans
        self.__defaultWordMan = defaultWordMan
        self.__addFeatures = addFeatures
        self.__vocab = vocab
        self.__classes = classes
        self.__defauthThreshold = defauthThreshold

        self.__transformToken = transformToken



    def __get_vocab(self):
        return self.__vocab

    vocab = property(__get_vocab)

    def __get_transformToken(self):
        return self.__transformToken


    def __set_transformToken(self, tt : Callable[[str] , str]):
        self.__transformToken = tt

    transformToken = property(__get_transformToken, __set_transformToken)
        
    def __getFeatures(self, tokens, text):

        features = []

        if self.__transformToken is None:
            for w in self.__vocab:
                features.append(1 if w in tokens else 0)
        else:
            ttokens = [self.__transformToken(t) for t in tokens] 
            for w in self.__vocab:
                if w in tokens:
                    features.append(1)
                else:
                    features.append(1 if self.__transformToken(w) in ttokens else 0)
        
        features.extend(self.__addFeatures(text, tokens))

        return np.array(features)

    def __tokensEquiv(self, tokens, maxInputs = 1000):
        res = []
    
        if "config" not in self.__jsonData: return res
        
        if "equivalent" not in self.__jsonData["config"]: return res
        
        cfgEquivs = self.__jsonData["config"]["equivalent"]

        nbTokens = len(tokens)
        
        tequivs = [None] * nbTokens

        iequivs = []

        for i, t in enumerate(tokens):
            if t in self.__vocab : continue

            for rightWord in cfgEquivs:
                if t in cfgEquivs[rightWord]:
                    if tequivs[i] is None:
                        tequivs[i] = [rightWord]
                        iequivs.append(i)
                    else:
                        tequivs[i].append(rightWord)
        
        w = 1
        for rw in tequivs:
            if rw is None or len(rw) < 2: continue
            
            w *= len(rw)

        nbEq = len(iequivs)
        if nbEq > 0:
            nbMax = (2 ** nbEq) if maxInputs is None else min(2 ** nbEq, maxInputs)

            permutations =  np.random.permutation(np.arange(1, nbMax))
            
            for p in permutations:

                antokens = [tokens.copy()]

                cbin = ("{0:0"+ str(nbEq) + "b}").format(p)
                
                for i, b in enumerate(cbin):
                    if b == '1':
                        eidx = iequivs[i]
                    
                        
                        neq = len(tequivs[eidx])
        

                        if neq > 1:

                            nantokens = []
                                
                            for ntokens in antokens:
                                if ntokens[eidx] in tequivs[eidx] : continue
                            
                                #print("ntokens", ntokens)
                                dtokens = []
                                
                                
                                for idt, eqw in enumerate(tequivs[eidx]):
                                    nd = ntokens.copy()
                                    nd[eidx] = eqw
                                    dtokens.append(nd)

                                nantokens.extend(dtokens)
                            
                            antokens = nantokens

                        else:

                            for ntokens in antokens:
                                if ntokens[eidx] in tequivs[eidx] : continue

                                for idt, eqw in enumerate(tequivs[eidx]):
                                    ntokens[eidx] = eqw
                    
                res.extend(antokens)

        return res
        
      
    def __inputScore(self, tokens, penalty = 0.9999):
    
        res = 1
        for t in tokens:
            if t in self.__vocab : continue
                
            res *= penalty
            
        return res

    def __buildInputs(self, tokens, maxInputs = 1000):
        res = [tokens]

        newInputs = self.__tokensEquiv(tokens, maxInputs)

        res.extend(newInputs)

        penalties = [self.__inputScore(tkns) for tkns in res]

        return res, penalties

    def answer(self, chatMan : ChatMan, text : str, threshold = None, noAnswerClass = "noAnswer"):
        if threshold is None: threshold = self.__defauthThreshold

        tokens = self.__defaultWordMan.tokenize(text)

        atokens, penalties = self.__buildInputs(tokens[0])

        result = []
        for idx, tokens in enumerate(atokens):
            features = self.__getFeatures(tokens, text)

            r = self.__fittedModel.predict(np.array([features]))[0]

            ajustedR = [ v * penalties[idx] for v in r]

            result.append(ajustedR)

        syntResult = [0.0] * len(self.__classes)
        for i in range(len(self.__classes)):
            for ar in result:
                if syntResult[i] < ar[i]: syntResult[i] = ar[i]
 
        y_pred = [[idx, res] for idx, res in enumerate(syntResult) if res > threshold]

        y_pred.sort(key=lambda x:x[1], reverse=True)

        selectedClasses = [self.__classes[r[0]] for r in y_pred]

        selectedClasses = chatMan.selectClasses(selectedClasses)

        if selectedClasses is None or len(selectedClasses) == 0:
            asw = chatMan.chooseAnswer(self.__jsonData, text, [noAnswerClass])
        else:
            asw = chatMan.chooseAnswer(self.__jsonData, text, selectedClasses)
            
        return asw

class WordTransformer:

    def __init__(self, accuracy, transform : Callable[[str], str]):
        self.__accuracy = accuracy
        self.__transform = transform


    def __get_accuracy(self):
        return self.__accuracy

    accuracy = property(__get_accuracy)

    def do(self, w: str):
        return self.__transform(w)


class ChatbotTrainer:

    #WT_IDENTITY = WordTransformer([0, 0], lambda str : str)

    #__defaultPatternMan = PatternMan()

    def __init__(self, defaultWordMan = DefaultWordManWithParams("fr_core_news_sm", "french", WordNetLemmatizer()), wordMans : Mapping[str, WordMan]=None, addFeatures : Callable[[str, Sequence[str]], Sequence[object]] = lambda _, __ : []):
        self.__defaultWordMan = defaultWordMan

        self.__addFeatures = addFeatures

        #self.__ET_STEM = WordTransformer([0, 1], lambda str : self.__defaultWordMan.stem(str))
        #self.__ET_LEM = WordTransformer([1, 0], lambda str : self.__defaultWordMan.lemmatize(str))

        if wordMans is None:
            self.__wordMans = dict()
            self.__wordMans["_default"] = defaultWordMan
        else:
            self.__wordMans = wordMans

    def initialize():
        nltk.download("punkt")
        nltk.download("wordnet")
        nltk.download('omw-1.4')

    def __estimator(self, inputShape, outputShape):
        model = Sequential()
        model.add(Dense(128, input_shape=inputShape, activation="relu"))
        model.add(Dropout(0.4))
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
                    pm = self.__defaultWordMan
                    tokens = pm.tokenize(pattern)

                    vocab.extend(tokens[0])
                    doc_X.append((pattern, tokens[0]))
                    doc_Y.append(tag)

                else:
                    pm = self.__wordMans[pattern['man']]
                    newPattern = pattern['pattern']

                    if isinstance(newPattern, str):
                        tokens = pm.tokenize(newPattern)

                        vocab.extend(tokens[0])
                        doc_X.append((newPattern, tokens[0]))
                        doc_Y.append(tag)
                    else:
                        for p in newPattern:
                            tokens = pm.tokenize(p)

                            vocab.extend(tokens[0])
                            doc_X.append((p, tokens[0]))
                            doc_Y.append(tag)
                
            if tag not in classes:
                classes.append(tag)
        
        #print(vocab)
        vocab = [w for w in vocab if w not in string.punctuation]
        
        #vocab = [self.__defaultWordMan.lemmatize(w) for w in vocab] + [self.__defaultWordMan.stem(w) for w in vocab]

        vocab = sorted(set(vocab))
        classes = sorted(set(classes))

        training = []
        outEmpty = [0] * len(classes)

        for idx, doc in enumerate(doc_X):
            features = []
            ftokens = doc[1]

            for w in vocab:
                features.append(1 if w in ftokens else 0)

            features.extend(self.__addFeatures(doc[0], ftokens))

            outputRow = list(outEmpty)
            outputRow[classes.index(doc_Y[idx])] = 1

            training.append([features, outputRow])
        

        random.shuffle(training)
        training = np.array(training, dtype=object)

        train_X = np.array(list(training[:, 0]))

        train_Y = np.array(list(training[:, 1]))

        inputShape = (len(train_X[0]),)
        outputShape = (len(train_Y[0]))

        model = self.__estimator(inputShape, outputShape)

        model.fit(x=train_X, y=train_Y, epochs=epochs)


        return Chatbot(jsonData, vocab, classes, self.__wordMans, self.__defaultWordMan, self.__addFeatures, model, threshold)


