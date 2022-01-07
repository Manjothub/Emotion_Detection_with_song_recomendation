# import important libraries
import os
import cv2
import numpy as np
from keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import matplotlib.pyplot as plt
import soundfile as sf
import sounddevice as sd
import random


play=None
# music list




#load model
model = load_model("best_model.h5")

face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# url='http://192.168.29.223:8080/video'
cap = cv2.VideoCapture(0)

# cap.open(url)



while True:
    ret, test_img = cap.read()
    if not ret:
        continue
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)
    
    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_img, (x,y), (x + w, y + h), (255, 0, 0), thickness=5)
        roi_gray = gray_img[y:y + w, x:x + h] #cropping region of interest i.e. face area from image
        roi_gray = cv2.resize(roi_gray, (224, 224))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        
        predictions = model.predict(img_pixels)
        max_index = np.argmax(predictions[0])
        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]
        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        music=['Music/1.wav','Music/2.wav','Music/3.wav','Music/4.wav','Music/5.wav','Music/6.wav']
        if  predicted_emotion in ['angry','sad','fear']:
            for i in random.choices(music):
                
                data, fs = sf.read(i, dtype='float32')
                sd.play(data, fs)
        else:
            sd.stop() 
            
    
    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial emotion analysis ', resized_img)

    
    if cv2.waitKey(27) == ord('q'): 
        break


cap.release()
cv2.destroyAllWindows()
                                    
         