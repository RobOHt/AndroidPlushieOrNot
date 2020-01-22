import numpy as np
import os
import cv2
import random
import pickle

DATADIR = "./data"
CATEGORIES = ["android_plushie", "non_android_plushie"]
img_size = 100
training_data = []
X = []
Y = []


def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)  # Get path in (plushie or non-plushie)
        class_num = CATEGORIES.index(category)  # Give each category name 1 or 0
        for img in os.listdir(path):
            try:
                unscaled_img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # Read img into array
                scaled_img_array = cv2.resize(unscaled_img_array, (img_size, img_size))
                training_data.append([scaled_img_array, class_num])
            except:
                print("faulty img index: ", str(os.listdir(path)))


create_training_data()
random.shuffle(training_data)
for features, label in training_data:
    X.append(features)
    Y.append(label)
X = np.array(X).reshape(-1, img_size, img_size, 1)


# load out
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("Y.pickle", "wb")
pickle.dump(Y, pickle_out)
pickle_out.close()

# load back in:
# pickle_in = open("X.pickle", "rb")
# X = pickle_out.load(pickle_in)
