import cv2
import math
import time
import argparse
import mediapipe as mp
import numpy as np

class AgeGenderDetector:
    def getFaceBox(net, frame,conf_threshold = 0.75):
        frameOpencvDnn = frame.copy()
        frameHeight = frameOpencvDnn.shape[0]
        frameWidth = frameOpencvDnn.shape[1]
        blob = cv2.dnn.blobFromImage(frameOpencvDnn,1.0,(300,300),[104, 117, 123], True, False)

        net.setInput(blob)
        detections = net.forward()
        bboxes = []

        for i in range(detections.shape[2]):
            confidence = detections[0,0,i,2]
            if confidence > conf_threshold:
                x1 = int(detections[0,0,i,3]* frameWidth)
                y1 = int(detections[0,0,i,4]* frameHeight)
                x2 = int(detections[0,0,i,5]* frameWidth)
                y2 = int(detections[0,0,i,6]* frameHeight)
                bboxes.append([x1,y1,x2,y2])
                cv2.rectangle(frameOpencvDnn,(x1,y1),(x2,y2),(0,255,0),int(round(frameHeight/150)),8)

        return frameOpencvDnn , bboxes



    faceProto = "body/opencv_face_detector.pbtxt"
    faceModel = "body/opencv_face_detector_uint8.pb"

    ageProto = "body/age_deploy.prototxt"
    ageModel = "body/age_net.caffemodel"

    genderProto = "body/gender_deploy.prototxt"
    genderModel = "body/gender_net.caffemodel"

    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']


    #load the network
    ageNet = cv2.dnn.readNet(ageModel,ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)
    faceNet = cv2.dnn.readNet(faceModel, faceProto)

    cap = cv2.VideoCapture(0)
    padding = 20

    hand = mp.solutions.hands
    Hand = hand.Hands(max_num_hands=2) #variável responsável por detectar a mão no video
    mpDraw = mp.solutions.drawing_utils #variável que desenha os pontos na mão

    while cv2.waitKey(1) < 0:
        #read frame
        t = time.time()
        hasFrame , frame = cap.read()

        if not hasFrame:
            cv2.waitKey()
            break
        #creating a smaller frame for better optimization
        small_frame = cv2.resize(frame,(0,0),fx = 0.5,fy = 0.5)

        frameFace ,bboxes = getFaceBox(faceNet,small_frame)
        if not bboxes:
            print("No face Detected, Checking next frame")
            continue
        for bbox in bboxes:
            face = small_frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),
                    max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]
            print("Gender : {}, conf = {:.3f}".format(gender, genderPreds[0].max()))

            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]
            print("Age Output : {}".format(agePreds))
            print("Age : {}, conf = {:.3f}".format(age, agePreds[0].max()))

            label = "{},{}".format(gender, age)
            cv2.putText(frameFace, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            

        chek, img = cap.read()
        imgRBG = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = Hand.process(imgRBG)
        handsPoins = results.multi_hand_landmarks #extração das coordenadas dos pontos
        h,w,_ = img.shape
        pontos = []
        if handsPoins: #quando a mão aparecer, a imagem roda
            for points in handsPoins:
                print(points)
                mpDraw.draw_landmarks(img, points, hand.HAND_CONNECTIONS) #DESENHO DOS PONTOS NA MÃO
                for id, cord in enumerate(points.landmark): 
                    cx, cy = int(cord.x*w), int(cord.y*h)
                        #cv2.putText(img,str(id),(cx,cy+10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)
                        #pontos.append((cx,cy))
            
        # Redimensione a imagem para que tenha as mesmas dimensões que frameFace
        img_resized = cv2.resize(img, (frameFace.shape[1], frameFace.shape[0]))

        # Misture as imagens de rosto e mão usando cv2.addWeighted()
        alpha = 0.5  # Ajuste o valor de alpha conforme necessário para controlar a intensidade da mistura
        beta = (1.0 - alpha)
        combined_frame = cv2.addWeighted(frameFace, alpha, img_resized, beta, 0.0)

        # Mostre a imagem combinada
        cv2.imshow("Age Gender Demo", combined_frame)


detector = AgeGenderDetector()
detector.run()



        
        