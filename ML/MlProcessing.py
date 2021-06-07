import numpy as np
from sklearn.model_selection import train_test_split


class SplitDataset:
    _testData = {}
    _trainData = {}

    def __init__(self, dataset=None, howMuch=None):
        self.dataset = dataset
        self.howMuch = howMuch

    def split(self):
        train_x, test_x, train_y, test_y = train_test_split(self.dataset[0], self.dataset[1], test_size=20,
                                                            random_state=42)
        self._trainData['train_x'] = train_x
        self._trainData['train_y'] = train_y
        self._testData['test_x'] = test_x
        self._testData['test_y'] = test_y

    def __getTrain(self):
        return self._trainData

    def __getTest(self):
        return self._testData


class TrainModel:

    def __init__(self, classifier=None, datasetx=None, datasety=None):
        self.classifier = classifier
        self.datasetx = datasetx
        self.datasety = datasety

    def train(self):
        if self.classifier is None or self.datasetx is None or self.datasety is None:
            return 1
        model = self.classifier.train(self.datasetx, self.datasety)
        return model

class metrics:

    _model = None
    _datatest = None

    def __init__(self, model=None, datatest=None):
        self._model = model
        self._datatest = datatest

    def modelScore(self):
        predicted = self.model.predict(self._datatest['test_x'])
        score = self.model.score(predicted, self._datatest['test_y'])

    def classificationMetrics(self):
        #import
        return None

