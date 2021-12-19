import pickle
import numpy
import tensorflow
from keras import Input, Model
from keras.initializers.initializers_v1 import RandomNormal
from keras.layers import Flatten, TimeDistributed, Dense, Dropout

from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.regularizers import l2

from data import get_data, get_train_data
from rcnn.config import Config


import tensorflow as tf
from tensorflow.keras.layers import Layer


print("Done!")