from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from django.views import View
from django.shortcuts import render
from django.http import HttpResponseRedirect, HttpResponse
from ML import MlProcessing as Ml
from api.views import managefiles
from statistic.models import statistic as statModel
import pickle
import time
from os import listdir
from os.path import isfile, join


@method_decorator(csrf_exempt, name='dispatch')
class statistic(View):
    pathModels = "./training/models/"
    datasetsPath = "./data/"

    def post(self, request):
        if request.POST.get('action'):
            # ~~~~~~~~~~~~~~~~~~~~~~~~ TRAIN ALGO ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
            if request.POST.get('action') == "train":
                from os import listdir
                from os.path import isfile, join
                training = None
                model = None
                metric = None
                template = None
                score = None
                onlyfiles = [f for f in listdir(self.datasetsPath) if isfile(join(self.datasetsPath, f))]
                files = managefiles(onlyfiles)
                filesTrain = [f for f in files if 'train' in f]
                testsizer = int(request.POST.get('testsize'))
                df = Ml.FilesManager.loadDataSet(self, fullpath=self.datasetsPath + filesTrain[1])
                split_data = Ml.SplitDataset(df)
                split_data.split(testSize=testsizer, RandomStat=42)
                if request.POST.get('algo') == "RandomForestClassifier100":
                    from sklearn.ensemble import RandomForestClassifier
                    training = Ml.TrainModel(RandomForestClassifier(n_estimators=100), split_data.dataset["Questions"],
                                             split_data.dataset["Reponses"])
                    model = training.train()
                    metric = Ml.metrics(model, split_data.getTest())
                    score = metric.modelScore()
                    tuple_objects = (model, score)
                    # Save tuple
                    pickle.dump(tuple_objects,
                                open(self.pathModels + request.POST.get('algo') + "_tuple_model.pkl", 'wb'))

                    # Restore tuple
                    statModel.objects.filter(algoName=request.POST.get('algo')).update(
                        algoName=request.POST.get('algo'),
                        trainSize=len(split_data.dataset['Questions']),
                        testSize=len(split_data.getTest()['test_x']),
                        Score=metric.modelScore(),
                        NLPmethode='ngram_range=(2, 2)',
                        details=self.pathModels + request.POST.get('algo') + "_tuple_model.pkl"
                    )

                    template = "pages/trainAndPrediction.html"
                if request.POST.get('algo') == "RandomForestClassifier200":
                    from sklearn.ensemble import RandomForestClassifier
                    training = Ml.TrainModel(RandomForestClassifier(n_estimators=200), split_data.dataset["Questions"],
                                             split_data.dataset["Reponses"])
                    model = training.train()
                    metric = Ml.metrics(model, split_data.getTest())
                    score = metric.modelScore()
                    tuple_objects = (model, score)
                    # Save tuple
                    pickle.dump(tuple_objects,
                                open(self.pathModels + request.POST.get('algo') + "_tuple_model.pkl", 'wb'))

                    # Restore tuple
                    statModel.objects.filter(algoName=request.POST.get('algo')).update(
                        algoName=request.POST.get('algo'),
                        trainSize=len(split_data.dataset['Questions']),
                        testSize=len(split_data.getTest()['test_x']),
                        Score=metric.modelScore(),
                        NLPmethode='ngram_range=(2, 2)',
                        details=self.pathModels + request.POST.get('algo') + "_tuple_model.pkl"
                    )

                    template = "pages/trainAndPrediction.html"
                if request.POST.get('algo') == "RandomForestClassifier50":
                    from sklearn.ensemble import RandomForestClassifier
                    training = Ml.TrainModel(RandomForestClassifier(n_estimators=50), split_data.dataset["Questions"],
                                             split_data.dataset["Reponses"])
                    model = training.train()
                    metric = Ml.metrics(model, split_data.getTest())
                    score = metric.modelScore()
                    tuple_objects = (model, score)
                    # Save tuple
                    pickle.dump(tuple_objects,
                                open(self.pathModels + request.POST.get('algo') + "_tuple_model.pkl", 'wb'))

                    # Restore tuple
                    statModel.objects.filter(algoName=request.POST.get('algo')).update(
                        algoName=request.POST.get('algo'),
                        trainSize=len(split_data.dataset['Questions']),
                        testSize=len(split_data.getTest()['test_x']),
                        Score=metric.modelScore(),
                        NLPmethode='ngram_range=(2, 2)',
                        details=self.pathModels + request.POST.get('algo') + "_tuple_model.pkl"
                    )

                    template = "pages/trainAndPrediction.html"
                if request.POST.get('algo') == "KNeighborsClassifier5":
                    from sklearn.neighbors import KNeighborsClassifier
                    training = Ml.TrainModel(KNeighborsClassifier(n_neighbors=5), split_data.dataset["Questions"],
                                             split_data.dataset["Reponses"])
                    model = training.train()
                    metric = Ml.metrics(model, split_data.getTest())
                    score = metric.modelScore()
                    tuple_objects = (model, score)
                    # Save tuple
                    pickle.dump(tuple_objects,
                                open(self.pathModels + request.POST.get('algo') + "_tuple_model.pkl", 'wb'))

                    # Restore tuple
                    statModel.objects.filter(algoName=request.POST.get('algo')).update(
                        algoName=request.POST.get('algo'),
                        trainSize=len(split_data.dataset['Questions']),
                        testSize=len(split_data.getTest()['test_x']),
                        Score=metric.modelScore(),
                        NLPmethode='ngram_range=(2, 2)',
                        details=self.pathModels + request.POST.get('algo') + "_tuple_model.pkl"
                    )

                    template = "pages/trainAndPrediction.html"
                if request.POST.get('algo') == "KNeighborsClassifier3":
                    from sklearn.neighbors import KNeighborsClassifier
                    training = Ml.TrainModel(KNeighborsClassifier(n_neighbors=3), split_data.dataset["Questions"],
                                             split_data.dataset["Reponses"])
                    model = training.train()
                    metric = Ml.metrics(model, split_data.getTest())
                    score = metric.modelScore()
                    tuple_objects = (model, score)
                    # Save tuple
                    pickle.dump(tuple_objects,
                                open(self.pathModels + request.POST.get('algo') + "_tuple_model.pkl", 'wb'))

                    # Restore tuple
                    statModel.objects.filter(algoName=request.POST.get('algo')).update(
                        algoName=request.POST.get('algo'),
                        trainSize=len(split_data.dataset['Questions']),
                        testSize=len(split_data.getTest()['test_x']),
                        Score=metric.modelScore(),
                        NLPmethode='ngram_range=(2, 2)',
                        details=self.pathModels + request.POST.get('algo') + "_tuple_model.pkl"
                    )

                    template = "pages/trainAndPrediction.html"
                if request.POST.get('algo') == "MultinomialNB":
                    from sklearn.naive_bayes import MultinomialNB
                    training = Ml.TrainModel(MultinomialNB(), split_data.dataset["Questions"],
                                             split_data.dataset["Reponses"])
                    model = training.train()
                    metric = Ml.metrics(model, split_data.getTest())
                    score = metric.modelScore()
                    tuple_objects = (model, score)
                    # Save tuple
                    pickle.dump(tuple_objects,
                                open(self.pathModels + request.POST.get('algo') + "_tuple_model.pkl", 'wb'))

                    # Restore tuple
                    statModel.objects.filter(algoName=request.POST.get('algo')).update(
                        algoName=request.POST.get('algo'),
                        trainSize=len(split_data.dataset['Questions']),
                        testSize=len(split_data.getTest()['test_x']),
                        Score=metric.modelScore(),
                        NLPmethode='ngram_range=(2, 2)',
                        details=self.pathModels + request.POST.get('algo') + "_tuple_model.pkl"
                    )

                    template = "pages/trainAndPrediction.html"
                if request.POST.get('algo') == "BernoulliNB":
                    from sklearn.naive_bayes import BernoulliNB
                    training = Ml.TrainModel(BernoulliNB(), split_data.dataset["Questions"],
                                             split_data.dataset["Reponses"])
                    model = training.train()
                    metric = Ml.metrics(model, split_data.getTest())
                    score = metric.modelScore()
                    tuple_objects = (model, score)
                    # Save tuple
                    pickle.dump(tuple_objects,
                                open(self.pathModels + request.POST.get('algo') + "_tuple_model.pkl", 'wb'))

                    # Restore tuple
                    statModel.objects.filter(algoName=request.POST.get('algo')).update(
                        algoName=request.POST.get('algo'),
                        trainSize=len(split_data.dataset['Questions']),
                        testSize=len(split_data.getTest()['test_x']),
                        Score=metric.modelScore(),
                        NLPmethode='ngram_range=(2, 2)',
                        details=self.pathModels + request.POST.get('algo') + "_tuple_model.pkl"
                    )

                    template = "pages/trainAndPrediction.html"
                if request.POST.get('algo') == "CategoricalNB":
                    from sklearn.naive_bayes import CategoricalNB
                    training = Ml.TrainModel(CategoricalNB(), split_data.dataset["Questions"],
                                             split_data.dataset["Reponses"])
                    model = training.train()
                    metric = Ml.metrics(model, split_data.getTest())
                    score = metric.modelScore()
                    tuple_objects = (model, score)
                    # Save tuple
                    pickle.dump(tuple_objects,
                                open(self.pathModels + request.POST.get('algo') + "_tuple_model.pkl", 'wb'))

                    # Restore tuple
                    statModel.objects.filter(algoName=request.POST.get('algo')).update(
                        algoName=request.POST.get('algo'),
                        trainSize=len(split_data.dataset['Questions']),
                        testSize=len(split_data.getTest()['test_x']),
                        Score=metric.modelScore(),
                        NLPmethode='ngram_range=(2, 2)',
                        details=self.pathModels + request.POST.get('algo') + "_tuple_model.pkl"
                    )

                    template = "pages/trainAndPrediction.html"
                if request.POST.get('algo') == "DecisionTreeClassifier":
                    from sklearn.tree import DecisionTreeClassifier
                    training = Ml.TrainModel(DecisionTreeClassifier(), split_data.dataset["Questions"],
                                             split_data.dataset["Reponses"])
                    model = training.train()
                    metric = Ml.metrics(model, split_data.getTest())
                    score = metric.modelScore()
                    tuple_objects = (model, score)
                    # Save tuple
                    pickle.dump(tuple_objects,
                                open(self.pathModels + request.POST.get('algo') + "_tuple_model.pkl", 'wb'))

                    # Restore tuple
                    statModel.objects.filter(algoName=request.POST.get('algo')).update(
                        algoName=request.POST.get('algo'),
                        trainSize=len(split_data.dataset['Questions']),
                        testSize=len(split_data.getTest()['test_x']),
                        Score=metric.modelScore(),
                        NLPmethode='ngram_range=(2, 2)',
                        details=self.pathModels + request.POST.get('algo') + "_tuple_model.pkl"
                    )

                    template = "pages/trainAndPrediction.html"
                if request.POST.get('algo') == "LogisticRegression":
                    from sklearn.linear_model import LogisticRegression
                    training = Ml.TrainModel(LogisticRegression(multi_class='multinomial', solver='lbfgs'),
                                             split_data.dataset["Questions"],
                                             split_data.dataset["Reponses"])
                    model = training.train()
                    metric = Ml.metrics(model, split_data.getTest())
                    score = metric.modelScore()
                    tuple_objects = (model, score)
                    # Save tuple
                    pickle.dump(tuple_objects,
                                open(self.pathModels + request.POST.get('algo') + "_tuple_model.pkl", 'wb'))

                    # Restore tuple
                    statModel.objects.filter(algoName=request.POST.get('algo')).update(
                        algoName=request.POST.get('algo'),
                        trainSize=len(split_data.dataset['Questions']),
                        testSize=len(split_data.getTest()['test_x']),
                        Score=metric.modelScore(),
                        NLPmethode='ngram_range=(2, 2)',
                        details=self.pathModels + request.POST.get('algo') + "_tuple_model.pkl"
                    )

                    template = "pages/trainAndPrediction.html"
                if request.POST.get('algo') == "SVC":
                    from sklearn.svm import SVC
                    training = Ml.TrainModel(SVC(), split_data.dataset["Questions"],
                                             split_data.dataset["Reponses"])
                    model = training.train()
                    metric = Ml.metrics(model, split_data.getTest())
                    score = metric.modelScore()
                    tuple_objects = (model, score)
                    # Save tuple
                    pickle.dump(tuple_objects,
                                open(self.pathModels + request.POST.get('algo') + "_tuple_model.pkl", 'wb'))

                    # Restore tuple
                    statModel.objects.filter(algoName=request.POST.get('algo')).update(
                        algoName=request.POST.get('algo'),
                        trainSize=len(split_data.dataset['Questions']),
                        testSize=len(split_data.getTest()['test_x']),
                        Score=metric.modelScore(),
                        NLPmethode='ngram_range=(2, 2)',
                        details=self.pathModels + request.POST.get('algo') + "_tuple_model.pkl"
                    )

                    template = "pages/trainAndPrediction.html"
                    # data = apimodel.objects.all()
                if request.POST.get('algo') == "SGDClassifier":
                    from sklearn.linear_model import SGDClassifier
                    training = Ml.TrainModel(SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42),
                                             split_data.dataset["Questions"],
                                             split_data.dataset["Reponses"])
                    model = training.train()
                    metric = Ml.metrics(model, split_data.getTest())
                    score = metric.modelScore()
                    tuple_objects = (model, score)
                    # Save tuple
                    pickle.dump(tuple_objects,
                                open(self.pathModels + request.POST.get('algo') + "_tuple_model.pkl", 'wb'))

                    # Restore tuple
                    statModel.objects.filter(algoName=request.POST.get('algo')).update(
                        algoName=request.POST.get('algo'),
                        trainSize=len(split_data.dataset['Questions']),
                        testSize=len(split_data.getTest()['test_x']),
                        Score=metric.modelScore(),
                        NLPmethode='ngram_range=(2, 2)',
                        details=self.pathModels + request.POST.get('algo') + "_tuple_model.pkl"
                    )

                    template = "pages/trainAndPrediction.html"
                    # data = apimodel.objects.all()

                algos = statModel.objects.all()
                trainedlist = statModel.objects.filter(Score__isnull=False)
                # prompt is a context variable that can have different values      depending on their context
                prompt = {
                    'score': score,
                    'algos': algos,
                    'trained': trainedlist,
                }
                template = "pages/trainAndPrediction.html"

                # GET request returns the value of the data with the specified key.
                return render(request, template, prompt)
            # ~~~~~~~~~~~~~~~~~~~~~~~~ PREDICT QUESTION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
            if request.POST.get('action') == "predict":
                obj = statModel.objects.get(algoName=request.POST.get('algo'))
                pickled_model, pickled_score = pickle.load(open(obj.details, 'rb'))
                score = pickled_score
                start_time = time.time()
                prediction = pickled_model.predict([request.POST.get('question')])
                end_time = time.time()
                algos = statModel.objects.all()
                trained = statModel.objects.filter(Score__isnull=False)
                # declaring template
                template = "pages/trainAndPrediction.html"
                # prompt is a context variable that can have different values      depending on their context
                prompt = {
                    'score': score,
                    'prediction': prediction,
                    'timeelapsed': end_time - start_time,
                    'algos': algos,
                    'trained': trained,

                }
                return render(request, template, prompt)
            # ~~~~~~~~~~~~~~~~~~~~~~~~ PREDICT ON FILE  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
            if request.POST.get('action') == "predictList":
                from os import listdir
                from os.path import isfile, join
                datatest = request.POST.get('datatest')
                obj = statModel.objects.get(algoName=request.POST.get('algo'))
                pickled_model, pickled_score = pickle.load(open(obj.details, 'rb'))
                onlyfiles = [f for f in listdir(self.datasetsPath) if isfile(join(self.datasetsPath, f))]
                files = managefiles(onlyfiles)
                dataset = [f for f in files if 'train' in f]
                df_dataset = Ml.FilesManager().loadDataSet(fullpath=self.datasetsPath + dataset[0][1])
                df_datatest = Ml.FilesManager().loadTestData(fullpath=self.datasetsPath + datatest)
                testClass = Ml.MakeTest(pickled_model, df_datatest, df_dataset)
                resultdata = testClass.doTest('Questions individu')
                template = "pages/testOnFile.html"
                onlyfiles = [f for f in listdir(self.datasetsPath) if isfile(join(self.datasetsPath, f))]
                trained = statModel.objects.filter(Score__isnull=False)
                files = managefiles(onlyfiles)
                filesTest = [f for f in files if 'test' in f]
                # prompt is a context variable that can have different values      depending on their context
                prompt = {
                    'file': datatest,
                    'moyenne': resultdata[0],
                    'tempsderep': resultdata[1],
                    'table': resultdata[2].to_dict('split'),
                    'trained': trained,
                    'files': filesTest,
                }
                return render(request, template, prompt)
            # ~~~~~~~~~~~~~~~~~~~~~~~~ ADD ALGO ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
            if request.POST.get('action') == "newalgo":
                _, created = statModel.objects.update_or_create(
                    algoName=request.POST.get('algo'),
                )
            return HttpResponseRedirect("/api/callbot/?action=uploadDataset")
        return HttpResponse('Unauthorized', status=401)
    def get(self, request):
        if request.GET.get("action"):
            # ~~~~~~~~~~~~~~~~~~~~~~~~ TRAIN PAGE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
            if request.GET.get("action") == "train":
                template = "pages/trainAndPrediction.html"
                # data = apimodel.objects.all()
                # prompt is a context variable that can have different values      depending on their context
                algos = statModel.objects.all()
                trained = statModel.objects.filter(Score__isnull=False)
                prompt = {
                    'algos': algos,
                    'trained': trained,
                }
                # GET request returns the value of the data with the specified key.
                return render(request, template, prompt)
            # ~~~~~~~~~~~~~~~~~~~~~~~~ TEST PAGE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
            if request.GET.get("action") == "test":
                template = "pages/testOnFile.html"
                onlyfiles = [f for f in listdir(self.datasetsPath) if isfile(join(self.datasetsPath, f))]
                trained = statModel.objects.filter(Score__isnull=False)
                files = managefiles(onlyfiles)
                filesTest = [f for f in files if 'test' in f]
                prompt = {
                    'trained': trained,
                    'files': filesTest,
                }
                # GET request returns the value of the data with the specified key.
                return render(request, template, prompt)
            template = "pages/dashboard.html"
            return HttpResponseRedirect('/models/statistic/')

        template = "pages/tables.html"
        # prompt is a context variable that can have different values      depending on their context
        data = statModel.objects.all()
        prompt = {
            'data': data,
        }
        # GET request returns the value of the data with the specified key.
        return render(request, template, prompt)

