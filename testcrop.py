import cv2
import time
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf

facetracker = load_model('facetracker.h5')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
while cap.isOpened():
    time.sleep(1)
    ret, frame = cap.read()
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(rgb, (256, 256))
    
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
        cv2.rectangle(frame, (abs_xmin, abs_ymin), (abs_xmax, abs_ymax), (255, 0, 0), 2)
        # Controls the label rectangle
        cv2.rectangle(frame, 
                      tuple(np.add(np.multiply(sample_coords[:2], [720,720]).astype(int), 
                                    [0,-30])),
                      tuple(np.add(np.multiply(sample_coords[:2], [720,720]).astype(int),
                                    [80,0])), 
                            (255,0,0), -1)
        
        # Controls the text rendered
        cv2.putText(frame, 'face', tuple(np.add(np.multiply(sample_coords[:2], [720,720]).astype(int),
                                               [0,-5])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    

    cropped_frame = frame[abs_ymin:abs_ymax, abs_xmin:abs_xmax]
    cv2.imshow('EyeTrack', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()