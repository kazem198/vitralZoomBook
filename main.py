import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

detector = HandDetector(detectionCon=0.8)
startDist = None
cx, cy = 500, 500
scale = 0


while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img1 = cv2.imread("cvarduino.jpg")
    hands, img = detector.findHands(img, draw=True)

    if len(hands) == 2:
        hand1 = hands[0]
        hand2 = hands[1]
        fingers1 = detector.fingersUp(hand1)
        fingers2 = detector.fingersUp(hand2)
        # print(fingers1)
        if fingers1 == [1, 1, 0, 0, 0] and fingers2 == [1, 1, 0, 0, 0]:
            lmList1 = hand1["lmList"]
            lmList2 = hand2["lmList"]

            if startDist == None:
                # length, info, img = detector.findDistance(lmList1[8][0:2], lmList2[8][0:2], img, color=(255, 0, 0),
                #                                           scale=10)
                length, info, img = detector.findDistance(hand1["center"], hand2["center"], img, color=(255, 0, 0),
                                                          scale=10)
                startDist = length

            # length, info, img = detector.findDistance(lmList1[8][0:2], lmList2[8][0:2], img, color=(255, 0, 0),
            #                                           scale=10)
            length, info, img = detector.findDistance(hand1["center"], hand2["center"], img, color=(255, 0, 0),
                                                      scale=10)

            scale = int(length-startDist)//2
            cx, cy = info[4:]
            print(scale)
        else:
            startDist = None

    try:
        h1, w1, _ = img1.shape
        newHeight, newWidth = ((h1+scale)//2)*2, ((w1+scale)//2)*2
        img1 = cv2.resize(img1, (newHeight, newWidth))
        #     img[10:260, 10:260] = img1
        img[cy-newHeight//2:cy+newHeight//2,
            cx-newWidth//2:cx+newWidth//2] = img1
    except:
        pass

    cv2.imshow("img", img)

    if cv2.waitKey(1) & 0xFf == ord("q"):
        cv2.destroyAllWindows()
        break
