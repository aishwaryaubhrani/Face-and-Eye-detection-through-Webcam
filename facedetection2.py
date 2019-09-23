import cv2
import numpy as np
import time

face_cascade = cv2.CascadeClassifier(r"C:\opencv\build\etc\haarcascades\haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(r"C:\opencv\build\etc\haarcascades\haarcascade_eye.xml")

cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
while True:
    ret, img = cap.read()
    ret =True
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 3)
        roi_gray = gray_image[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (255, 255, 0), 3)

    cv2.imshow('img',img)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
   
cap.release()
cv2.destroyAllWindows()
