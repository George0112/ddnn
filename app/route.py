import numpy as np
import json
import requests
from flask import request
import ast
import logging
from app.my_model import my_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from flask import jsonify

from app import app

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def init(model_name, cut_point, next_cut_point, is_first=False, is_last=False, output_layer=-1, num_output=1):
    model = my_model(model_name, cut_point=cut_point, next_cut_point=next_cut_point, is_first=is_first, is_last=is_last)
    
    @app.route('/', methods=['POST', 'GET'])
    def post():
        input = request.form['data']
        input = ast.literal_eval(input)
        input = np.array(input)
        output = model.predict(input).tolist()

        if not is_last:
            if len(next_cut_point) == 1:
                try:
                    res = requests.post('http://'+model_name+'-'+str(next_cut_point[0]+1)+'.default.svc.cluster.local:5000', data = {'data': json.dumps(output)}).text
                except Exception as e:
                    print(e)
                    res = 'Network error'
                return res
            else:
                res = []
                for n in next_cut_point[1:]:
                    print(n)
                    try:
                        res.append(requests.post('http://'+model_name+'-'+str(n)+'.default.svc.cluster.local:5000', data = {'data': json.dumps(output)}).text)
                        print(res)
                    except Exception as e:
                        print(e)
                        res.append('Network error')
                        continue
                return '\n'.join(str(r) for r in res)
        else:
            index = np.argmax(output)
            return str(np.argmax(output))

    @app.route('/info', methods=['GET'])
    def info():
        res = []
        res.append(model.get_layers())
        if not is_last:
            if len(next_cut_point) == 1:
                try:
                    for l in json.loads(requests.get('http://'+model_name+'-'+str(next_cut_point[0]+1)+'.default.svc.cluster.local:5000/info').text):
                        res.append(l)
                    print(res)
                except Exception as e:
                    print(e)
            else:
                for n in next_cut_point[1:]:
                    print(n)
                    try:
                        for l in json.loads(requests.get('http://'+model_name+'-'+str(n)+'.default.svc.cluster.local:5000/info').text):
                            res.append(l)
                        print(res)
                    except Exception as e:
                        print(e)
                        continue
            return jsonify(res)
        else:
            return jsonify([model.get_layers()])

    @app.route('/name', methods=['GET'])
    def name():
        return model_name

    @app.route('/cuttable', methods=['GET'])
    def cuttable():
        if not is_first:
            return None
        else:
            return jsonify(model.get_cuttable())
