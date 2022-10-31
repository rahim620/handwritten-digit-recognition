import tensorflow as tf
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('../handwritten.model')

image_number = 1
while os.path.isfile(f"digits/{image_number}.png"):
    try:
        img = cv2.imread(f"digits/{image_number}.png")[:, :, 0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"{np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error!")
    finally:
        image_number += 1
