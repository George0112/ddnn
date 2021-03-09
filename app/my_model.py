from cut_dnn import cut_model_functional
from flask import jsonify

class my_model():
    def __init__(self, model_name, cut_point, next_cut_point, is_first=False, is_last=False):
        print(model_name, cut_point, next_cut_point, (is_first), is_last)

        self.cuttable = []

        model = self.pick_model(model_name)
        model.summary()
        
        for idx, layer in enumerate(model.layers[:-1]):
            if len(layer._inbound_nodes) > 1:
                continue
            else:
                self.cuttable.append({"index": idx, "name": layer.name})

        if is_first:
            self.model, _ = cut_model_functional(model, next_cut_point[0])
            self.model.summary()
        elif is_last:
            _, self.model = cut_model_functional(model, cut_point, output_layer=next_cut_point[0])
            self.model.summary()
        else:
            model1, _ = cut_model_functional(model, next_cut_point[0])
            _, self.model = cut_model_functional(model1, cut_point)
            self.model.summary()
        pass

    def predict(self, input):
        output = self.model.predict(input)
        return output

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