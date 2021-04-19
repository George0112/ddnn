import numpy as np
import json
import requests
from flask import request
import ast
import logging
from app.my_model import my_model
# from tensorflow.keras.preprocessing.image import load_img
# from tensorflow.keras.preprocessing.image import img_to_array
from flask import jsonify
import time
import io
import zlib

from app import app

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def compress_nparr(nparr):
    """
    Returns the given numpy array as compressed bytestring,
    the uncompressed and the compressed byte size.
    """
    bytestream = io.BytesIO()
    np.save(bytestream, nparr)
    uncompressed = bytestream.getvalue()
    compressed = zlib.compress(uncompressed)
    print("compress from %f to %f" %(len(uncompressed), len(compressed)))
    return compressed

def uncompress_nparr(bytestring):
    """
    """
    return np.load(io.BytesIO(zlib.decompress(bytestring)))

def init(model_name, cut_point, next_cut_point, is_first=False, is_last=False, output_layer=-1, num_output=1):
    model = my_model(model_name, cut_point=cut_point, next_cut_point=next_cut_point, is_first=is_first, is_last=is_last)
    
    @app.route('/', methods=['POST', 'GET'])
    def post():
        start = time.time()
        input = request.get_data()
        input = uncompress_nparr(input)
        # print("Request from form time usage: %f" %(time.time()-start))
        # input = ast.literal_eval(input)
        # print("literal_eval time usage: %f" %(time.time()-start))
        # input = np.array(input)
        print("Data retrieve time usage: %f" %(time.time()-start))
        output = model.predict(input)

        if not is_last:
            if len(next_cut_point) == 1:
                try:
                    res = requests.post('http://'+model_name+'-'+str(next_cut_point[0]+1)+'.default.svc.cluster.local:5000', data=compress_nparr(output)).text
                except Exception as e:
                    print(e)
                    res = 'Network error'
                return res
            else:
                res = []
                for n in next_cut_point[1:]:
                    print(n)
                    try:
                        res.append(requests.post('http://'+model_name+'-'+str(n)+'.default.svc.cluster.local:5000', data=compress_nparr(output)).text)
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
        if not is_last:
            res = []
            res.append(model.get_layers())
            if len(next_cut_point) == 1:
                try:
                    for l in json.loads(requests.get('http://'+model_name+'-'+str(next_cut_point[0]+1)+'.default.svc.cluster.local:5000/info').text):
                        res.append(l)
                except Exception as e:
                    print(e)
                    return 'model not ready', 500
            else:
                for n in next_cut_point[1:]:
                    print(n)
                    try:
                        for l in json.loads(requests.get('http://'+model_name+'-'+str(n)+'.default.svc.cluster.local:5000/info').text):
                            res.append(l)
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

    @app.route('/time', methods=['GET'])
    def get_avg_time():
        if not is_last:
            res = []
            res.append(model.get_avg_time())
            if len(next_cut_point) == 1:
                try:
                    for l in json.loads(requests.get('http://'+model_name+'-'+str(next_cut_point[0]+1)+'.default.svc.cluster.local:5000/time').text):
                        res.append(l)
                except Exception as e:
                    print(e)
            else:
                for n in next_cut_point[1:]:
                    print(n)
                    try:
                        for l in json.loads(requests.get('http://'+model_name+'-'+str(n)+'.default.svc.cluster.local:5000/time').text):
                            res.append(l)
                    except Exception as e:
                        print(e)
                        continue
            return jsonify(res)
        else:
            return jsonify([model.get_avg_time()])

    @app.route('/layer', methods=['POST'])
    def change_layer():
        cut_point = int(request.form['cut_point'])
        next_cut_point = [int(n) for n in request.form['next_cut_point'].split(',')]
        # try:
        model = my_model(model_name, cut_point=cut_point, next_cut_point=next_cut_point, is_first=is_first, is_last=is_last)
        # except Exception as e:
            # print(e)
        
        return jsonify({'cut_point': cut_point, 'next_cut_point': next_cut_point})



    @app.route('/metrics', methods=['GET'])
    def get_metric():
        if not is_last:
            res = []
            res.append({"time": model.get_time(), "cpu": model.get_cpu(), "memory": model.get_memory()})
            if len(next_cut_point) == 1:
                try:
                    for l in json.loads(requests.get('http://'+model_name+'-'+str(next_cut_point[0]+1)+'.default.svc.cluster.local:5000/metrics').text):
                        res.append(l)
                except Exception as e:
                    print(e)
            else:
                for n in next_cut_point[1:]:
                    print(n)
                    try:
                        for l in json.loads(requests.get('http://'+model_name+'-'+str(n)+'.default.svc.cluster.local:5000/metrics').text):
                            res.append(l)
                    except Exception as e:
                        print(e)
                        continue
            return jsonify(res)
        else:
            return jsonify([{"time": model.get_time(), "cpu": model.get_cpu(), "memory": model.get_memory()}])
