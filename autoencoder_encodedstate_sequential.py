'''
  Visualizing the encoded state of a simple autoencoder created with the Keras Sequential API
  with Keract.
'''
import keras
from keras.layers import Dense
from keras.datasets import mnist
from keras.models import Sequential
from keract import get_activations, display_activations
import matplotlib.pyplot as plt
from keras import backend as K

# Model configuration
img_width, img_height = 28, 28
initial_dimension = img_width * img_height
batch_size = 128
no_epochs = 1
validation_split = 0.2
verbosity = 1
encoded_dim = 50

# Load MNIST dataset
(input_train, target_train), (input_test, target_test) = mnist.load_data()

# Reshape data
input_train = input_train.reshape(input_train.shape[0], initial_dimension)
input_test = input_test.reshape(input_test.shape[0], initial_dimension)
input_shape = (initial_dimension, )

# Parse numbers as floats
input_train = input_train.astype('float32')
input_test = input_test.astype('float32')

# Normalize data
input_train = input_train / 255
input_test = input_test / 255

# Define the 'autoencoder' full model
autoencoder = Sequential()
autoencoder.add(Dense(encoded_dim, activation='relu', kernel_initializer='he_normal', input_shape=input_shape))
autoencoder.add(Dense(initial_dimension, activation='sigmoid'))

# Compile the autoencoder
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Give us some insights
autoencoder.summary()

# Fit data
autoencoder.fit(input_train, input_train, epochs=no_epochs, batch_size=batch_size, validation_split=validation_split)

# =============================================
# Take a sample for visualization purposes
# =============================================
input_sample = input_test[:1]
reconstruction = autoencoder.predict([input_sample])

# =============================================
# Visualize input-->reconstruction
# =============================================
fig, axes = plt.subplots(1, 2)
fig.set_size_inches(6, 3.5)
input_sample_reshaped = input_sample.reshape((img_width, img_height))
reconsstruction_reshaped = reconstruction.reshape((img_width, img_height))
axes[0].imshow(input_sample_reshaped) 
axes[0].set_title('Original image')
axes[1].imshow(reconsstruction_reshaped)
axes[1].set_title('Reconstruction')
plt.show()

# =============================================
# Visualize encoded state with Keract
# =============================================
activations = get_activations(autoencoder, input_sample)
display_activations(activations, cmap="gray", save=False)