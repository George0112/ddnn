import numpy as np
import json
import requests
import time
import logging
import argparse
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('model', type=str)
parser.add_argument('input_size', type=int)

args = parser.parse_args()

first_part = args.model + '-0'

while True:
    image = load_img('./cat.jpg', target_size=(args.input_size, args.input_size))
    image = img_to_array(image)
    image = np.array([image for x in range(1)]).tolist()
    # res = requests.post('http://'+first_part+'.default.svc.cluster.local:5000', data = {'data': json.dumps(image)}).text
    res = requests.post('http://tesla.cs.nthu.edu.tw:32510/', data = {'data': json.dumps(image)}).text
    print(res)
    logger.info(res)
    time.sleep(10)