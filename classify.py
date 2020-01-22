import numpy
import plaidml.keras
plaidml.keras.install_backend()
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time

NAME = "Plushie_or_Not_{time}".format(time=int(time.time()))
tensorboard = TensorBoard(log_dir='logs/{name}'.format(name=NAME))

X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

X = X/255
y = numpy.array(y)

model = Sequential()

model.add(Conv2D(256, (2, 2), input_shape=X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(512))
model.add(Activation("relu"))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

model.fit(X, y, batch_size=32, epochs=20, validation_split=0.05, callbacks=[tensorboard])

model.save(f'{NAME}.model')

# Save into tensorflow lite model
converter = tf.lite.TFLiteConverter.from_saved_model(f'{NAME}.model')
tflite_model = converter.convert()
open(f"{NAME}.tflite", "wb").write(tflite_model)
