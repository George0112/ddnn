import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import plot_model
import numpy as np

layer_outputs = {}

def get_output_of_layer(layer, starting_layer_name, new_input, output_layer=-1):
    
    # if we have already applied this layer on its input(s) tensors,
    # just return its already computed output
    if layer.name in layer_outputs:
        return layer_outputs[layer.name]

    # if this is the starting layer, then apply it on the input tensor
    if layer.name == starting_layer_name:
    # print(layer._inbound_nodes[0].inbound_layers)
    # if starting_layer_name == layer._inbound_nodes[0].inbound_layers.name:
        # layer._inbound_nodes.pop()
        out = layer(new_input)
        layer_outputs[layer.name] = out
        return out

    # find all the connected layers which this layer
    # consumes their output
    prev_layers = []
    for node in layer._inbound_nodes:
        prev_layers.append(node.inbound_layers)

    # get the output of connected layers
    pl_outs = []
    for pl in prev_layers:
        pl_outs.extend([get_output_of_layer(pl, starting_layer_name, new_input)])

    # apply this layer on the collected outputs
    out = layer(pl_outs[0] if len(pl_outs) == 1 else pl_outs)
    layer_outputs[layer.name] = out
    return out

# Functional models
def cut_model_functional(model, cut_point, output_layer=-1):
    
    global layer_outputs
    try:
        starting_layer_name = model.layers[cut_point].name
        print(starting_layer_name)
        # create a new input layer  for our sub-model we want to construct
        layer_input_shape = model.get_layer(starting_layer_name).get_input_shape_at(0)
        new_input = layers.Input(batch_shape=layer_input_shape)
        if isinstance(layer_input_shape, list):
            layer_input = [layers.Input(shape=layer_input_shape[0][1:]) for x in range(len(layer_input_shape))]
        else:
            layer_input = layers.Input(shape=layer_input_shape[1:])
        new_input = layer_input
        layer_outputs = {}

        in_l = model.get_layer(index=0)
        out_l = model.get_layer(index=cut_point)

        modelA = Model(inputs=in_l.get_input_at(0), outputs=out_l.get_output_at(0))
        
        new_output = get_output_of_layer(model.layers[output_layer], starting_layer_name, new_input)

        # create the sub-model
        modelB = Model(inputs=new_input, outputs=new_output)

    except Exception as e:
        print(e)
        for layer in model.layers:
            if len(layer._inbound_nodes) > 1:
                layer._inbound_nodes.pop()
        raise Exception('Multiple inbound nodes')

    for layer in model.layers:
        if len(layer._inbound_nodes) > 1:
            layer._inbound_nodes.pop()
    
    for layer in modelB.layers:
        if len(layer._inbound_nodes) > 1:
            layer._inbound_nodes.pop()

    return modelA, modelB
