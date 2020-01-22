import cv2
import tensorflow as tf
from dataPrep import CATEGORIES

model = tf.keras.models.load_model("Plushie_or_Not_1579708065.model")


def getImg(PATH):
    size = 100
    unscaled_img_array = cv2.imread(PATH, cv2.IMREAD_GRAYSCALE)
    scaled_img_array = cv2.resize(unscaled_img_array, (size, size))
    return scaled_img_array.reshape(-1, size, size, 1)


while True:
    FILENAME = input('Input a file name please.')
    prediction = model.predict([getImg(FILENAME)])  # input is a list
    print("This is a {}".format(CATEGORIES[int(prediction[0, 0])]))
