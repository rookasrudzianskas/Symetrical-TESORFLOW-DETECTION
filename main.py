import ssl
import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

#  permissions to download from online
# ssl._create_default_https_context = ssl._create_unverified_context
#  lOading predefined dataset
fashion_mnist = keras.datasets.fashion_mnist
# pulling out from the database
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print(train_labels[0])

plt.imshow(train_images[0], cmap='gray', vmin=0, vmax=255)
plt.show()
