'''
  ConvNet filter visualization with Activation Maximization on exemplary VGG16 Keras model
'''
from keras.applications import VGG16
from vis.utils import utils
from vis.visualization import visualize_activation, get_num_filters
from vis.input_modifiers import Jitter
import matplotlib.pyplot as plt
import numpy as np
import random
import os.path

# Define the folder name to save into
folder_name = 'filter_visualizations'

# Define the model
model = VGG16(weights='imagenet', include_top=True)

# Iterate over multiple layers
for layer_nm in ['block1_conv1', 'block2_conv1', 'block3_conv2', 'block4_conv1', 'block5_conv2']:

  # Find the particular layer
  layer_idx = utils.find_layer_idx(model, layer_nm)

  # Get the number of filters in this layer
  num_filters = get_num_filters(model.layers[layer_idx])

  # Draw 6 filters randomly
  drawn_filters = random.choices(np.arange(num_filters), k=6)

  # Visualize each filter
  for filter_id in drawn_filters:
    img = visualize_activation(model, layer_idx, filter_indices=filter_id, input_modifiers=[Jitter(16)])
    plt.imshow(img)
    img_path = os.path.join('.', folder_name, layer_nm + '_' + str(filter_id) + '.jpg')
    plt.imsave(img_path, img)
    print(f'Saved layer {layer_nm}/{filter_id} to file!')