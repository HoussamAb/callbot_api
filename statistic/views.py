from django.http import JsonResponse
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from django.views import View
import csv, io
from django.shortcuts import render
from django.contrib import messages
import json  # Create your views here.
from ML import MlProcessing as Ml
from statistic.models import statistic as statModel
import pickle
from collections import namedtuple
import time



@method_decorator(csrf_exempt, name='dispatch')
class statistic(View):
    """def post(self, request):

        data = json.loads(request.body.decode("utf-8"))
        p_file = data.get('file_name')
        p_methode = data.get('methode')

        product_data = {
            'file_name': p_file,
            'methode': p_methode,
        }

        cart_item = CartItem.objects.create(**product_data)

        data = {
            "message": f"New item added to Cart with id: {cart_item.id}"
        }
        return JsonResponse(data, status=201)
"""

    def post(self, request):
        if request.POST.get('action'):
            if request.POST.get('action') == "train":
                training = None
                model = None
                metric = None
                template = None
                score = None
                df = Ml.FilesManager.loadDataSet(self, fullpath='./data/dataset.csv')
                split_data = Ml.SplitDataset(df)
                split_data.split(testSize=20, RandomStat=42)
                if request.POST.get('algo') == "RF":
                    from sklearn.ensemble import RandomForestClassifier
                    training = Ml.TrainModel(RandomForestClassifier(), split_data.dataset["Questions"],
                                             split_data.dataset["Reponses"])
                    model = training.train()
                    metric = Ml.metrics(model, split_data.getTest())
                    score = metric.modelScore()
                    tuple_objects = (model, score)
                    # Save tuple
                    pickle.dump(tuple_objects, open(request.POST.get('algo')+"_tuple_model.pkl", 'wb'))

                    # Restore tuple
                    _, created = statModel.objects.update_or_create(
                        algoName='RandomForestClassifier',
                        trainSize=len(split_data.dataset['Questions']),
                        testSize=len(split_data.getTest()['test_x']),
                        Score=metric.modelScore(),
                        NLPmethode='ngram_range=(2, 2)',
                        details=request.POST.get('algo')+"_tuple_model.pkl"
                    )

                    template = "pages/trainAndPrediction.html"
                # data = apimodel.objects.all()
                # prompt is a context variable that can have different values      depending on their context
                prompt = {
                    'order': 'Order of the CSV should be Questions, responses',
                    'profiles': None,
                    'score': score
                }
                # GET request returns the value of the data with the specified key.
                return render(request, template, prompt)
        if request.POST.get('action'):
            if request.POST.get('action') == "predict":
                if request.POST.get('algo') == "RF":
                    obj = statModel.objects.get(algoName="RandomForestClassifier")
                    pickled_model, pickled_score = pickle.load(open(obj.details, 'rb'))
                    score = pickled_score
                    start_time = time.time()
                    prediction = pickled_model.predict([request.POST.get('question')])
                    end_time = time.time()

                    # declaring template
                    template = "pages/trainAndPrediction.html"
                    # prompt is a context variable that can have different values      depending on their context
                    prompt = {
                        'score':score,
                        'prediction':prediction,
                        'timeelapsed':end_time-start_time
                    }
                    return render(request, template, prompt)
        if request.POST.get('action'):
            if request.POST.get('action') == "predictList":
                None
        if request.POST.get('action'):
            if request.POST.get('action') == "newalgo":
                None

    def get(self, request):
        if request.GET.get("action"):
            if request.GET.get("action") == "train":
                template = "pages/trainAndPrediction.html"
                # data = apimodel.objects.all()
                # prompt is a context variable that can have different values      depending on their context
                prompt = {
                    'order': 'Order of the CSV should be Questions, responses',
                    'profiles': None,
                    'trained': None
                }
                # GET request returns the value of the data with the specified key.
                return render(request, template, prompt)
            # ~~~~~~~~~~~~~~~~~~~~~~~~ UPLOAD PAGE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
            if request.GET.get("action") == "uploadDataset":
                template = "pages/uploadFiles.html"
                # data = apimodel.objects.all()
                # prompt is a context variable that can have different values      depending on their context
                prompt = {
                    'order': 'Order of the CSV should be Questions, responses',
                    'profiles': None,
                }
                # GET request returns the value of the data with the specified key.
                return render(request, template, prompt)

            template = "csv_loader.html"
            # prompt is a context variable that can have different values      depending on their context
            prompt = {
                'order': 'Order of the CSV should be Questions, responses',
            }
            return render(request, template, prompt)

        template = "pages/tables.html"
        # data = apimodel.objects.all()
        # prompt is a context variable that can have different values      depending on their context
        prompt = {
            'order': 'Order of the CSV should be Questions, responses',
            'profiles': None,
        }
        # GET request returns the value of the data with the specified key.
        return render(request, template, prompt)
