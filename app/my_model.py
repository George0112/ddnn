from cut_dnn import cut_model_functional
import tensorflow as tf
from flask import jsonify
from tensorflow import keras
import gc
import time
import threading
import psutil

class my_model():
    def __init__(self, model_name, cut_point, next_cut_point, is_first=False, is_last=False):
        print(model_name, cut_point, next_cut_point, is_first, is_last)

        self.cuttable = []
        self.avg_time = (0, 0)
        self.cpu = []
        self.times = []
        self.memory = []
        tf.keras.backend.clear_session()

        model = self.pick_model(model_name)

        if cut_point == 0 and next_cut_point[0] == 0:
            self.model = model
            return
        
        for idx, layer in enumerate(model.layers[:-1]):
            if len(layer._inbound_nodes) > 1:
                continue
            else:
                self.cuttable.append({"index": idx, "name": layer.name})

        if is_first:
            model, _ = cut_model_functional(model, next_cut_point[0])
            model.save('./model')
            tf.keras.backend.clear_session()
            del model
            gc.collect()
            self.model = keras.models.load_model('./model')
            self.model.summary()
        elif is_last:
            _, self.model = cut_model_functional(model, cut_point, output_layer=next_cut_point[0])
            self.model.summary()
        else:
            model1, _ = cut_model_functional(model, next_cut_point[0])
            _, self.model = cut_model_functional(model1, cut_point)
            self.model.summary()
        pass

    def record(self):
        global running
        running = True
        currentProcess = psutil.Process()
        # start loop
        while running:
            self.memory.append(currentProcess.memory_info().rss)
            self.cpu.append(currentProcess.cpu_percent(interval=0.01))

    def start(self):
        global t
        # create thread and start it
        t = threading.Thread(target=self.record)
        t.start()

    def stop(self):
        global running
        global t
        # use `running` to stop loop in thread so thread will end
        running = False
        # wait for thread's end
        t.join()

    def predict(self, input):
        self.start()
        try:
            start = time.time()
            output = self.model.predict(input)
            t = time.time() - start
            self.times.append(t)
            print("predict time usage: %f" %(t))
            self.avg_time = ((self.avg_time[0]*self.avg_time[1] + t)/(self.avg_time[1]+1), self.avg_time[1]+1)
            return output
        finally:
            self.stop()

    def pick_model(self, model_name):
        if 'vgg16' in model_name:
            from tensorflow.keras.applications import VGG16
            return VGG16(weights="imagenet")
        elif 'multitask' in model_name:
            from models.multitask import multitask
            return multitask
        else:
            raise Exception('Unsupportted model')
    
    def get_layers(self):
        return([
            {
                "name": l.name, 
                "input": l.input.name, 
                "input_shape": str(l.input_shape),
                "output_shape": str(l.output_shape)
            } for l in self.model.layers])

    def get_cuttable(self):
        return self.cuttable

    def get_avg_time(self):
        return self.avg_time[0]

    def get_time(self):
        return self.times

    def get_cpu(self):
        return self.cpu

    def get_memory(self):
        return self.memory