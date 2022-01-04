from flask import Flask,render_template,Response
import cv2
import dlib
from imutils import face_utils
import numpy as np
import vlc
import time

app=Flask(__name__)
cap = cv2.VideoCapture(0)


faceDetector=dlib.get_frontal_face_detector()
landmarkDetector=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def calculateDistance(ptA,ptB):
    return np.linalg.norm(ptA-ptB)

def isBlinked(a,b,c,d,e,f):
    smallDistance=calculateDistance(b,d)+calculateDistance(c,e)
    largeDistance=calculateDistance(a,f)

    ratio=smallDistance/(2*largeDistance)

    if ratio>=0.25:
        return "ACTIVE"

    elif ratio<0.25 and ratio>=0.21:
        return "DROWSY"
    
    else:
        return "SLEEPY"

def generateFrames():
    global sleepy,active,drowsy,status,color
    
    while True:
        _,frame=cap.read()

        if(_==False):
            return ""
        
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceDetector(grayFrame)
        
        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
            
            # cv2.rectangle(faframe, (x1, y1), (x2, y2), (0, 255, 0), 2)
            landmarks = landmarkDetector(grayFrame, face)
            landmarks = face_utils.shape_to_np(landmarks)


            left_blink = isBlinked(landmarks[36],landmarks[37], 
                    landmarks[38], landmarks[41], landmarks[40], landmarks[39])
            right_blink = isBlinked(landmarks[42],landmarks[43], 
                    landmarks[44], landmarks[47], landmarks[46], landmarks[45])

            if(left_blink=="SLEEPY" or right_blink=="SLEEPY"):
                sleepy+=1
                drowsy=0
                active=0
                if(sleepy>=6):
                    status="SLEEPY"
                    color=(0,0,255)
            elif(left_blink=="DROWSY" or right_blink=="DROWSY"):
                sleepy=0
                drowsy+=1
                active=0
                if(drowsy>=6):
                    status="DROWSY"
                    color=(0,255,0)
            else:
                sleepy=0
                drowsy=0
                active+=1
                if(active>=1):
                    status="ACTIVE"
                    color=(255,0,0)
            
            
            if(status!="ACTIVE"):
                p = vlc.MediaPlayer("mixkit-facility-alarm-908.wav")
                p.play()
                start = time.time()
                # print("hello")
                while(1):
                    end = time.time()
                    if(end-start>2):
                        break
                # print(end - start)

            cv2.putText(frame, status, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color,3)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        ret,buffer=cv2.imencode('.jpg',frame)
        frame=buffer.tobytes()
        
        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():

    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generateFrames(),mimetype='multipart/x-mixed-replace; boundary=frame')#,index()

if __name__ == "__main__":
    SLEEPY = 0
    drowsy = 0
    active = 0
    status="No face Detected"
    color=(0,0,0)
    app.run(debug=True)
