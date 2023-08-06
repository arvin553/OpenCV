import cv2
import mediapipe as mp
import time

cap=cv2.VideoCapture(0) #取得電腦鏡頭畫面
mpHands=mp.solutions.hands #使用手部模型
hands=mpHands.Hands(min_detection_confidence=0.9,min_tracking_confidence=0.9) #嚴謹度
mpDraw=mp.solutions.drawing_utils
handLmStyle=mpDraw.DrawingSpec(color=(0,0,255),thickness=5)  #手掌偵測點的樣式設定
handConStyle=mpDraw.DrawingSpec(color=(0,255,255),thickness=10) #線的樣式
pTime=0 #previous time
cTime=0 #current time

while True:
    ret,img= cap.read()
    if ret:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        faceCascade = cv2.CascadeClassifier('face_detect.xml')  # opencv github上訓練好的人臉辨識模型
        faceRect = faceCascade.detectMultiScale(gray, 1.1, 5)  # 偵測的圖片,縮小的比例去偵測有沒有人臉,這張臉至少要被框到幾次才算是真的被偵測到

        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        result=hands.process(imgRGB)

        for (x, y, w, h) in faceRect:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        imgHeight=img.shape[0] #獲得視窗高度
        imgWidth=img.shape[1]  #獲得視窗寬度



        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:  #偵測到的每一隻手
                mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS,handLmStyle,handConStyle)
                for i,lm in enumerate(handLms.landmark): #第i個點,landmark第i個點的座標
                    xPos=int(lm.x * imgWidth)
                    yPos=int(lm.y* imgHeight)

                    if i==4:
                        cv2.circle(img,(xPos,yPos),20,(200,150,50),cv2.FILLED)
                    print(i,xPos,yPos)
        cTime=time.time()
        fps= 1/(cTime-pTime)
        pTime=cTime
        cv2.putText(img,f"FPS:{int(fps)}",(30,50),cv2.FONT_HERSHEY_TRIPLEX,1,[0,255,0],3)

        cv2.imshow('img',img)
    if cv2.waitKey(1)== ord('q'):
        break