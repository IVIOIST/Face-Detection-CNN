import cv2
import time
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import face_recognition
import ctypes


# Tensorflow Setup
GPU = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(GPU[0], True)
except:
    print('failed to set memory growth on GPU')
    pass

face_detector = load_model('VGG19_REV1.h5')

# Face_Recognition Setup
known_faces = []
for i in range(1, 9):
    my_face = face_recognition.load_image_file(os.path.join('face_data', 'known', f'face{i}.jpg'))
    my_face_encoding = face_recognition.face_encodings(my_face)
    if len(my_face_encoding) > 0:
        known_faces.append(my_face_encoding[0])
    else:
        print(f"No face detected in image face{i}.jpg")

# OpenCV Setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Declaring Variables
counter = 0
last_reset_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(rgb, (224, 224))
    yhat = face_detector.predict(np.expand_dims(resized/255, 0))

    xmin, ymin, xmax, ymax = yhat[1][0]
    abs_xmin, abs_ymin, abs_xmax, abs_ymax = np.multiply([xmin, ymin, xmax, ymax], 720).astype(int)
    
    if yhat[0] > 0.5: 
        cv2.rectangle(frame, (abs_xmin - 20, abs_ymin - 20), (abs_xmax - 50, abs_ymax - 50), (255,0,0), 2)
        cv2.circle(frame, (abs_xmin - 20, abs_ymin - 20), radius=2, color=(0, 0, 255), thickness=2)
        cv2.circle(frame, (abs_xmax - 50, abs_ymax - 50), radius=2, color=(0, 0, 255), thickness=2)


        try:
            cropped_frame = frame[abs_ymin - 20:abs_ymax - 50, abs_xmin - 20:abs_xmax - 50]
            rgb_cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
            unknown_face_encoding = face_recognition.face_encodings(rgb_cropped_frame)
            result = face_recognition.compare_faces(my_face_encoding, unknown_face_encoding)
            flag = False
            for known_face in known_faces:
                results = face_recognition.compare_faces(known_face, unknown_face_encoding, tolerance=0.5)
                if results[0] == True:  
                    counter = 0
                    flag = True
                    print(flag)
                    break
            if flag == False:
                counter += 1 
                print(flag)
        except:
            pass
        if counter == 3:
            print('fail limit reached')
            counter = 0
            ctypes.windll.user32.LockWorkStation()
        current_time = time.time()
        if current_time - last_reset_time > 60:
            counter = 0
            last_reset_time = current_time
    time.sleep(1)
    cv2.imshow('Real Time', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 

cap.release()
cv2.destroyAllWindows()
