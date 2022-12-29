#For data preparation and preprocessing
import tensorflow as tf
import tensorflow.keras
from keras.preprocessing.image import ImageDataGenerator
#from scipy.misc import toimage

#For model creation and training
from keras.models import Sequential
from keras.layers import Conv2D, AveragePooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping, ModelCheckpoint


class Model():
  
  def cnnModel():
    num_classes = 1
    model = Sequential([
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.AveragePooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.AveragePooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.AveragePooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation = 'sigmoid' ) #Output layer
        # tf.keras.layers.Dense(num_classes, activation = 'softmax' ) #Output layer
        ])

    model.build(input_shape = (None,256,256,3))
    model.summary()

    return model


#Model.cnnModel()    