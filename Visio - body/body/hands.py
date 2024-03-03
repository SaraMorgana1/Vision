import cv2
import mediapipe as mp

class hand:

    video = cv2.VideoCapture(0)

    hand = mp.solutions.hands
    Hand = hand.Hands(max_num_hands=2) #variável responsável por detectar a mão no video
    mpDraw = mp.solutions.drawing_utils #variável que desenha os pontos na mão

    while True:
        chek, img = video.read()
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
            


        cv2.imshow("Imagem", img)
        cv2.waitKey(2)    # podemos processar a imagem

dthand=hand()
dthand.run






