import cv2
import numpy as np
import argparse
net = cv2.dnn.readNetFromCaffe("model/model/MobileNetSSD_deploy.prototxt",
                               "model/model/MobileNetSSD_deploy10695.caffemodel")

blub = False    #默认红灯 False->红灯 True->绿灯
#获取人脸
def getFace(filename):   
    face_xml = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    img = cv2.imread(filename,1)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_xml.detectMultiScale(gray,1.3,5)
    #print('face=',len(faces))
    # draw
    index = 1
    if len(faces)>0:
        print(len(faces))
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_face = gray[y:y+h,x:x+w]
        roi_color = img[y:y+h,x:x+w]
        outname = str(index)+'.jpg'
        cv2.imwrite(outname,roi_color)
        index = index + 1
        img = cv2.imread(outname,1)
        cv2.imshow('dst',img)
        cv2.waitKey(0)
#绘制人脸
def draw_person(img, persont):
    x,y,w,h = persont
    cv2.rectangle(img,(x,y),((x+w),(y+h)-15),(255,255,0),2)
cap = cv2.VideoCapture('2.mp4')   #读取摄像头
while True:
    ret, image = cap.read()
    (h, w) = image.shape[:2]
    t1 = cv2.getTickCount()
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0 / 255,
                                 (300, 300), (104.0, 177.0, 123.0))#目标与行人相像的百分比
    net.setInput(blob)
    detections = net.forward()

    # 循环寻找行人踪迹
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.1:  # args["confidence"]:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY),
                          (0, 0, 255), 2)
            cv2.putText(image, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    t2 = cv2.getTickCount()
    t = (t2 - t1) / cv2.getTickFrequency()
    cv2.imshow("Output", image)
    if cv2.waitKey(20) & 0xff == ord('q'):
        break
#释放窗口
cap.release()
cv2.destroyAllWindows()
