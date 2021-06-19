from django.http import JsonResponse
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from api.models import api as apimodel
from django.views import View
import csv, io
from django.shortcuts import render
from django.contrib import messages
import jsonapi  # Create your views here.
import ML.MlProcessing as Ml
from statistic.models import statistic as Statmodel


def managefiles(files):
    filetuple = []
    for file in files:
        if file.lower().find('train') != -1:
            filetuple.append(['train', file, file[-3:]])
        elif file.lower().find('test') != -1:
            filetuple.append(['test', file, file[-3:]])
        else:
            filetuple.append(['unknown', file, file[-3:]])
    return filetuple


@method_decorator(csrf_exempt, name='dispatch')
class api(View):
    datasetsPath = './data/'
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
        # declaring template
        print("im here")
        template = "pages/uploadFiles.html"
        message = None
        # data = apimodel.objects.all()
        # prompt is a context variable that can have different values      depending on their context

        # GET request returns the value of the data with the specified key.
        # if request.method == "GET":
        #   return render(request, template, prompt)
        csv_file = request.FILES['file']
        print(csv_file.name)
        # let's check if it is a csv file
        if not csv_file.name.endswith('.csv'):
            prompt = {
                "messages": ('THIS IS NOT A CSV FILE')
            }
            return render(request, template, prompt)
        data_set = csv_file.read().decode('ansi')
        # setup a stream which is when we loop through each line we are able to handle a data in a stream
        if csv_file.name.lower().find('train') != -1:
            io_string = io.StringIO(data_set)
            next(io_string)
            with open('data/' + csv_file.name, 'w', encoding='ansi', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=';', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(["Questions", "Reponses"])
                for column in csv.reader(io_string, delimiter=';'):
                    writer.writerow([column[0], column[1]])
        elif csv_file.name.lower().find('test') != -1:
            io_string = io.StringIO(data_set)
            next(io_string)
            with open('data/' + csv_file.name, 'w', encoding='ansi', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=';', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(["Question", "Réponse Inbenta"])
                for column in csv.reader(io_string, delimiter=';'):
                    writer.writerow([column[0], column[1]])
        else:
            message = 'Error file type not accepted'

        prompt = {
            'messages': message,
        }

        # for column in csv.reader(io_string, delimiter=';'):
        #    _, created = apimodel.objects.update_or_create(
        #        questions=column[0],
        #        reponses=column[1],
        #    )
        context = {}
        return render(request, template, context)

    def get(self, request):
        if request.GET.get("action"):
            # ~~~~~~~~~~~~~~~~~~~~~~~~ UPLOAD PAGE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
            if request.GET.get("action") == "uploadDataset":
                template = "pages/uploadFiles.html"
                from os import listdir
                from os.path import isfile, join
                onlyfiles = [f for f in listdir(self.datasetsPath) if isfile(join(self.datasetsPath, f))]
                # data = apimodel.objects.all()
                # prompt is a context variable that can have different values      depending on their context
                print(managefiles(onlyfiles))
                data = Statmodel.objects.all()
                prompt = {
                    'files': managefiles(onlyfiles),
                    'algorithm': data,
                    'order': 'Order of the CSV should be Questions, responses',
                    'profiles': None,
                }
                # GET request returns the value of the data with the specified key.
                return render(request, template, prompt)

            if request.GET.get("action") == "datasets":
                template = "pages/datasets.html"
                from os import listdir
                from os.path import isfile, join
                onlyfiles = [f for f in listdir(self.datasetsPath) if isfile(join(self.datasetsPath, f))]
                files = managefiles(onlyfiles)
                dataset = [f for f in files if 'train' in f]
                df_dataset = Ml.FilesManager().loadDataSet(fullpath=self.datasetsPath + dataset[0][1])
                df_dataset_ = df_dataset.to_dict('split')
                print(df_dataset_)
                data = zip(df_dataset_['data'], df_dataset_['index'])
                prompt = {
                    'dataset': data,
                }
                # GET request returns the value of the data with the specified key.
                return render(request, template, prompt)
            template = "csv_loader.html"
            data = apimodel.objects.all()
            # prompt is a context variable that can have different values      depending on their context
            prompt = {
                'order': 'Order of the CSV should be Questions, responses',
                'profiles': data
            }
            return render(request, template, prompt)

        template = "pages/dashboard.html"
        # data = apimodel.objects.all()
        # prompt is a context variable that can have different values      depending on their context
        prompt = {
            'order': 'Order of the CSV should be Questions, responses',
            'profiles': None,
        }
        # GET request returns the value of the data with the specified key.
        return render(request, template, prompt)

