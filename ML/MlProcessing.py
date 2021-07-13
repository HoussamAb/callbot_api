import re
import string
import time

import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


class FilesManager:
    _testDataframe = None  # inbenta Test
    _allDataSet = None  # Dataset_allquetions

    def loadTestData(self, fullpath):
        self._testDataframe = pd.read_csv(fullpath, delimiter=";", encoding='ansi')
        return self._testDataframe
        # chargement de donnée a partir de fichier fullpath/filename.CSV

    def loadDataSet(self, fullpath):
        self._allDataSet = pd.read_csv(fullpath, delimiter=";", encoding='ansi')
        return self._allDataSet
#               NLP PROCESSING
class NlpProcessing:
    stopword = nltk.corpus.stopwords.words('french')  # All frensh Stopwords
    ps = nltk.PorterStemmer()
    wn = nltk.WordNetLemmatizer()

    def clean_text(text):
        text = "".join([word.lower() for word in text if word not in string.punctuation])
        tokens = re.split('\W+', text)
        # text = [ps.stem(word) for word in tokens if word not in stopword]
        # text = [wn.lemmatize(word) for word in tokens]
        return text
#               SPLIT DATA
class SplitDataset:
    _testData = {}
    _trainData = {}

    def __init__(self, dataset=None):
        if dataset is None:
            self.dataset = FilesManager.loadDataSet('data/dataset_train.csv')
        self.dataset = dataset

    def split(self, testSize, RandomStat):
        train_x, test_x, train_y, test_y = train_test_split(self.dataset["Questions"], self.dataset["Reponses"],
                                                            test_size=testSize,
                                                            random_state=RandomStat)
        self._trainData['train_x'] = train_x
        self._trainData['train_y'] = train_y
        self._testData['test_x'] = test_x
        self._testData['test_y'] = test_y

    def getTrain(self):
        return self._trainData

    def getTest(self):
        return self._testData
#                   TRAIN MODEL
class TrainModel:
    _countvectorizer = CountVectorizer(ngram_range=(2, 2), analyzer=NlpProcessing.clean_text)
    _tfidftransformer = TfidfTransformer()
    _classifier = None

    def __init__(self, classifier=None, datasetx=None, datasety=None):
        self._classifier = classifier
        self.datasetx = datasetx
        self.datasety = datasety

    def initPipeline(self):
        text_clf = Pipeline([('vect', self._countvectorizer),
                             ('tfidf', self._tfidftransformer),
                             ('clf', self._classifier), ])
        return text_clf

    def train(self):
        if self._classifier is None or self.datasetx is None or self.datasety is None:
            return 1
        pipe = self.initPipeline()
        model = pipe.fit(X=self.datasetx, y=self.datasety)
        return model
#                   BUILD METRICS
class Metrics:
    _model = None
    _datatest = None
    _predicted = None

    def __init__(self, model=None, datatest=None):
        self._model = model
        self._datatest = datatest

    def modelScore(self):
        self._predicted = self._model.predict(self._datatest['test_x'])
        # score = self.model.score(predicted, self._datatest['test_y'])
        score = np.mean(self._predicted == self._datatest['test_y']) * 100
        return score

    def classificationMetrics(self):
        if self._predicted is None:
            self.modelScore()
        from sklearn.metrics import classification_report
        report = classification_report(self._datatest['test_y'], self._predicted)
        return report

    def accuracy(self):
        from sklearn.metrics import accuracy_score
        if self._predicted is None:
            self._predicted = self._model.predict(self._datatest['test_x'])
        print("la precision sur valeurs prédites égale : %.3f %%" % (
                    accuracy_score(self._datatest['test_y'], self._predicted) * 100))
#                   Test Sur Fichier INBENTA
class MakeTest:
    df1 = None
    _algo = None
    df2 = None

    def __init__(self, algo, datatest, dataset):
        self._algo = algo
        self.df1 = datatest
        self.df2 = dataset

    def predict_statis(self, q):
        return self._algo.predict([q])

    def compare_dataframes(self, df1, df2):
        l1 = []
        l2 = []
        l3 = []
        l4 = []
        for i, rowi in df1.iterrows():
            for j, rowj in df2.iterrows():
                if (str(rowi['Réponse Inbenta']).lower() == str(rowj['Questions']).lower()):
                    l1.append(rowi['Question'])
                    l2.append(rowi['Réponse Inbenta'])
                    l3.append(rowj['Questions'])
                    l4.append(rowj['Reponses'])
        df = pd.DataFrame({
            'Questions individu': l1,
            'Question inbenta': l2,
            'Question vrai': l3,
            'Réponse vrai': l4
        })
        return df

    def comp(self, df):
        label = []
        for i, row in zip(df['réponce algorithme'], df['Réponse vrai']):
            str(i).replace("[", "").replace("]", "")
            if i == row:
                label.append(1)
            else:
                label.append(0)
        df['label'] = label

    def moyenneR(self, df):
        som = 0
        for i in df.label:
            if i == 1:
                som += 1
        return ((som / len(df)) * 100)

    def moyenne_tempsR(self, df):
        som = 0
        result = []
        numberoftest = len(df['temps de réponse (s)'])
        # print('Le totale des tests : ', len(df['temps de réponse (s)']))
        for i in df['temps de réponse (s)']:
            som += i
        result.append(numberoftest)
        result.append((som / len(df)))
        return result

    def moyenne(self, df):
        som = 0
        for i in df.label:
            if i == 1:
                som += 1
        return "Moyenne est : %.4f %%" % ((som / len(df)) * 100)

    def moyenne_temps(self, df):
        som = 0
        numberoftest = len(df['temps de réponse (s)'])
        # print('Le totale des tests : ', len(df['temps de réponse (s)']))
        for i in df['temps de réponse (s)']:
            som += i
        return "La Moyenne de temps de réponses : %.4f \n <br> Le totale des tests : %d" % (
        (som / len(df), numberoftest))

    # 'Questions individu' 'Question vrai'
    def doTest(self, colomnName):
        data = self.compare_dataframes(self.df1, self.df2)
        response = []
        test_rep = []
        temps_de_rep = []
        for i in data[colomnName]:
            start_time = time.time()
            test_rep.append(self.predict_statis([i]))
            temps_de_rep.append(time.time() - start_time)
        data['réponce algorithme'] = test_rep
        data['temps de réponse (s)'] = temps_de_rep
        self.comp(data)
        m = self.moyenneR(data)
        t = self.moyenne_tempsR(data)
        response.append(m)
        response.append(t)
        response.append(data)
        return response
