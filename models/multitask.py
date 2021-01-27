import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import os

multitask = models.Sequential()

input_layer = layers.Input(shape=(32, 32, 3))
block1_conv1 = layers.Conv2D(3, (3, 3), activation='relu', padding='SAME')(input_layer)
block1_conv2 = layers.Conv2D(32, (3, 3), activation='relu', padding='SAME')(block1_conv1)
block1_pool = layers.MaxPooling2D((2, 2))(block1_conv2)
block2_conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='SAME')(block1_pool)
block2_conv2 = layers.Conv2D(32, (3, 3), activation='relu', padding='SAME')(block2_conv1)
block2_pool = layers.MaxPooling2D((2, 2))(block2_conv2)
block3_conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='SAME')(block2_pool)
block3_conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='SAME')(block3_conv1)
block3_pool = layers.MaxPooling2D((2, 2))(block3_conv2)
block4_conv1 = layers.Conv2D(128, (3, 3), activation='relu', padding='SAME')(block3_pool)
block4_conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='SAME')(block4_conv1)
block4_conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='SAME')(block4_conv2)
block4_pool = layers.MaxPooling2D((2, 2))(block4_conv3)

# Smile branch
smile_fc1 = layers.Dense(256, activation='relu')(block4_pool)
smile_fc2 = layers.Dense(256, activation='relu')(smile_fc1)
smile_output = layers.Dense(2, activation='softmax')(smile_fc2)

# Gender branch
gender_fc1 = layers.Dense(256, activation='relu')(block4_pool)
gender_fc2 = layers.Dense(256, activation='relu')(gender_fc1)
gender_output = layers.Dense(2, activation='softmax')(gender_fc2)

# Age branch
age_fc1 = layers.Dense(256, activation='relu')(block4_pool)
age_fc2 = layers.Dense(256, activation='relu')(age_fc1)
age_output = layers.Dense(5, activation='softmax')(age_fc2)

multitask = Model(inputs=input_layer, outputs = [smile_output, gender_output, age_output])
