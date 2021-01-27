from cut_dnn import cut_model_functional
from tensorflow.keras.applications import VGG16
from models.multitask import multitask
import tensorflow as tf

print(tf.__version__)

class my_model():
    def __init__(self, model, cut_point, next_cut_point, is_first=False, is_last=False):
        print(model, cut_point, next_cut_point, (is_first), is_last)
        # model = VGG16(weights="imagenet")
        model = multitask
        if is_first:
            self.model, _ = cut_model_functional(model, next_cut_point[0])
            self.model.summary()
        elif is_last:
            _, self.model = cut_model_functional(model, cut_point)
            self.model.summary()
        else:
            model1, _ = cut_model_functional(model, next_cut_point[0])
            _, self.model = cut_model_functional(model1, cut_point)
            self.model.summary()
        pass
    def predict(self, input):
        output = self.model.predict(input)
        return output
