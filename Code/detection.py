#face, smile and eyes detection
import cv2
#import numpy as np
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
eyes_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        cv2.putText(img, 'face', (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 1, cv2.LINE_AA)
        rei_gray = gray[y:y+h, x:x+w] #roi=region of interest inside the rectangle of the face detect in the gray img
        rei_color = img[y:y+h, x:x+w]
        smile = smile_cascade.detectMultiScale(rei_gray,1.7,26)
        for (sx,sy,sw,sh) in smile: 
            cv2.rectangle(rei_color, (sx,sy), (sx+sw,sy+sh), (0,255,0), 2)
            cv2.putText(rei_color, 'smile', (sx, sy-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 1, cv2.LINE_AA)
        eyes = eyes_cascade.detectMultiScale(rei_gray,1.1,22)
        for (ex,ey,ew,eh) in eyes: 
            cv2.rectangle(rei_color, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)
            cv2.putText(rei_color, 'eyes', (ex, ey-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 1, cv2.LINE_AA)
    
    cv2.imshow('img',img)
    #cv2.imshow('gray',gray)
    key = cv2.waitKey(1)
    if key == 27:
        break
    
cap.release()  #closes the webcam
cv2.destroyAllWindows()   #closes the windows
