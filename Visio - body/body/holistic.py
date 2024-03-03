import cv2 
import mediapipe as mp

class detectorh:

    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

    cap=cv2.VideoCapture(0)

    with mp_holistic.Holistic(min_detection_confidence=-.5,min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame =cap.read()

            image=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = holistic.process(image)

            print(results.pose_landmarks)

            image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

            f1=mp_drawing.DrawingSpec(color=(80,110,10), thickness=1,circle_radius=1)
            f2=mp_drawing.DrawingSpec(color=(80,256,121), thickness=1,circle_radius=1)

            hr1=mp_drawing.DrawingSpec(color=(80,22,10), thickness=2,circle_radius=4)
            hr2=mp_drawing.DrawingSpec(color=(80,44,121), thickness=2,circle_radius=2)

            hl1=mp_drawing.DrawingSpec(color=(121,22,76), thickness=2,circle_radius=4)
            hl2=mp_drawing.DrawingSpec(color=(121,44,250), thickness=2,circle_radius=2)

            p1=mp_drawing.DrawingSpec(color=(245,117,66), thickness=2,circle_radius=4)
            p2=mp_drawing.DrawingSpec(color=(245,66,230), thickness=2,circle_radius=2)


            mp_drawing.draw_landmarks(image,results.face_landmarks,mp_holistic.FACEMESH_TESSELATION,f1,f2)

            mp_drawing.draw_landmarks(image,results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,hr1,hr2)
            
            mp_drawing.draw_landmarks(image,results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,hl1,hl2)
            
            mp_drawing.draw_landmarks(image,results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,p1,p2)
    
            cv2.imshow("holistic vision",image)

            if cv2.waitKey(10) & 0xFF == ord("e"):
                break


    cap.release()
    cv2.destroyAllWindows()

d=detectorh()
d.run()
