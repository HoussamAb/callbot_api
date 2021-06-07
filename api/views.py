from django.http import JsonResponse
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt

from api.models import api as apimodel
from django.views import View
import csv, io
from django.shortcuts import render
from django.contrib import messages
import json  # Create your views here.


@method_decorator(csrf_exempt, name='dispatch')
class api(View):
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
        template = "csv_loader.html"
        data = apimodel.objects.all()
        # prompt is a context variable that can have different values      depending on their context
        prompt = {
            'order': 'Order of the CSV should be name, email, address,    phone, profile',
            'profiles': data
        }
        # GET request returns the value of the data with the specified key.
        if request.method == "GET":
            return render(request, template, prompt)
        csv_file = request.FILES['file']
        # let's check if it is a csv file
        if not csv_file.name.endswith('.csv'):
            prompt = {
                "messages": ('THIS IS NOT A CSV FILE')
            }
            return render(request, template, prompt)
        data_set = csv_file.read().decode('ansi')
        # setup a stream which is when we loop through each line we are able to handle a data in a stream
        io_string = io.StringIO(data_set)
        next(io_string)
        for column in csv.reader(io_string, delimiter=';'):
            _, created = apimodel.objects.update_or_create(
                questions=column[0],
                reponses=column[1],
            )
        context = {}
        return render(request, template, context)

    def get(self, request):
        if request.GET.get("action"):
            template = "csv_loader.html"
            data = apimodel.objects.all()
            # prompt is a context variable that can have different values      depending on their context
            prompt = {
                'order': 'Order of the CSV should be Questions, responses',
                'profiles': data
            }
            return render(request, template, prompt)

        template = "csv_loader.html"
        # data = apimodel.objects.all()
        # prompt is a context variable that can have different values      depending on their context
        prompt = {
            'order': 'Order of the CSV should be Questions, responses',
            'profiles': None
        }
        # GET request returns the value of the data with the specified key.
        return render(request, template, prompt)
