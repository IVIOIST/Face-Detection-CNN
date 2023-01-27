import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np
import time
import os
import uuid

facetracker = load_model('VGG19_REV1.h5')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


while cap.isOpened():
    _ , frame = cap.read()
    time.sleep(1)
    IMAGE_NAME = os.path.join('face_data', f'{ str(uuid.uuid1())}.jpg')
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(rgb, (224, 224))
    yhat = facetracker.predict(np.expand_dims(resized/255,0))

    sample_coords = yhat[1][0]
    xmin = sample_coords[0]
    ymin = sample_coords[1]
    xmax = sample_coords[2]
    ymax = sample_coords[3]

    abs_xmin = np.multiply(xmin, 720).astype(int)
    abs_ymin = np.multiply(ymin, 720).astype(int)
    abs_xmax = np.multiply(xmax, 720).astype(int)
    abs_ymax = np.multiply(ymax, 720).astype(int)

    print(abs_xmin, abs_ymin, abs_xmax, abs_ymax)
    if yhat[0] > 0.5: 
        # Controls the main rectangle
        cv2.rectangle(frame, (abs_xmin - 20, abs_ymin - 20), (abs_xmax - 50, abs_ymax - 50), (255,0,0), 2)
        cv2.circle(frame, (abs_xmin - 20, abs_ymin - 20), radius=2, color=(0, 0, 255), thickness=2)
        cv2.circle(frame, (abs_xmax - 50, abs_ymax - 50), radius=2, color=(0, 0, 255), thickness=2)
    cv2.imshow('EyeTrack', frame)
    cropped_frame = frame[abs_ymin - 20:abs_ymax - 50, abs_xmin - 20:abs_xmax - 50]
    try:
        cv2.imwrite(IMAGE_NAME, cropped_frame)
    except cv2.error:
        pass
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
  