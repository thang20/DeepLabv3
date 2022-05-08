import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
from tqdm import tqdm
import cv2
from glob import glob
from sklearn.model_selection import train_test_split
from model import build_unet
H = 256
W = 256
num_classes = 4
def load_data(path, split = 0.1):
    X = sorted(glob(os.path.join(path, "image", "*.png")))
    Y = sorted(glob(os.path.join(path, "mask", "*.png")))
    print(len(X))
    print(len(Y))
    return X, Y

if __name__ == "__main__":
    """ Seeding """
    model = tf.keras.models.load_model("files/model.h5")
    x = cv2.imread('newdata/3benh/1.png', cv2.IMREAD_COLOR)
    x = cv2.resize(x, (W, H))
    x = x / 255.0
    x = x.astype(np.float32)

    ## Prediction
    p = model.predict(np.expand_dims(x, axis=0))[0]
    p = np.argmax(p, axis=-1)
    p = np.expand_dims(p, axis=-1)
    p = p.astype(np.int32)

    a1 = np.zeros(shape=(256, 256, 1))
    a1[p == 1] = 255
    a2 = np.zeros(shape=(256, 256, 1))
    a2[p == 2] = 255
    a3 = np.zeros(shape=(256, 256, 1))
    a3[p == 3] = 255

    b = np.zeros(shape=(256, 256, 1))
    b[p == 1] = 1
    b[p == 2] = 1
    b[p == 3] = 1


    p = np.concatenate([a1, a2, a3], axis=2)

    x = x * 255.0
    x = x.astype(np.int32)
    p2 = b * x

    h, w, _ = x.shape
    line = np.ones((h, 10, 3)) * 255


    final_image = np.concatenate([x, line, p, line, p2], axis=1)
    cv2.imwrite(f"results/1.png", final_image)



