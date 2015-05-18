import cv2
import numpy as np
import time
from matplotlib import pyplot as plt

# Code to parse video frame by frame

def cannyEdgeDetection(frame,p1,p2):
    return cv2.Canny(frame,200,300)

def getDeviation(points, h, k, r):
    err = 0
    for i in range(0, len(points)):
        _cX = points[i][0][0]
        _cY = points[i][0][1]
        #shift points
        cX = _cX
        cY = _cY
        if cX == 0:
            err += r
            continue
        # get the slope of line from center to the contour point
        if (h-cX) == 0:
            continue
            #line is vertical
        else:
            m = float(k - cY) / (h - cX)
        #y = mx + c
        cLine = cY - m*cX
        #get the corresponding point in circumference
        #  x^2 + m^2 * x^2 + 2 * m * x * cLine + cLine^2 = r^2
        # (1+m^2)*x^2 + 2*m*cLine*x + cLine^2 - r^2 = 0
        a = 1 + m * m
        b =  2 * m * (cLine - k) - 2 * h
        c = h*h + (cLine-k) * (cLine - k) - r * r
        det = b*b-4*a*c
        if det >= 0 :
            _x1 = (-b + np.sqrt(det) )/ 2*a
            _x2 = (-b - np.sqrt(det) )/ 2*a

            _y1 = np.sqrt(abs(r*r - (_x1-h)*(_x1-h))) + k
            _y2 = np.sqrt(abs(r*r - (_x2-h)*(_x2-h))) + k
            #print "({},{}), ({},{})".format(_x1, _y1, _x2, _y2)
            dis1 = (cX - _x1) * (cX - _x1) + (cY - _y1) * (cY - _y1)
            dis2 = (cX - _x2) * (cX - _x2) + (cY - _y2) * (cY - _y2)

            if(dis1 > dis2):
                err += dis2
            else:
                err += dis1

    return float(err) / len(points)

def getCroppedImage(img, x, y , radius):
    startX = x - 2*radius
    startY = y - 2*radius
    if startX < 0:
        startX = 0

    if startY < 0:
        startY = 0

    endX = startX + 4 * radius
    endY = startY + 4 * radius

    if endX > img.shape[0]:
        endX = img.shape[0]

    if endY > img.shape[1]:
        endY = img.shape[1]

    return img[startY:endY, startX:endX]

#based on paper
def hasSpeedLimitSign(frame):
    edge = cannyEdgeDetection(frame,150,250)
    (rows, column) = edge.shape

    draw = np.zeros(rows, dtype=np.int8)

    t = cv2.findContours(edge, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    contours = [cv2.approxPolyDP(cnt, 3, True) for cnt in t[0]]

    speedLimit = frame
    error = 100000
    for i in range(0,len(contours)):
        if(len(contours[i]) > 5):
            center, radius = cv2.minEnclosingCircle(contours[i])
            if radius < frame.shape[0]/8 :
                x, y = center
                x = int(x)
                y = int(y)
                
                new_error = getDeviation(contours[i], x, y, radius)
                #print  "center : ({},{})  radius: {}  error: {}".format(x, y, radius ,error)
                if new_error < error:
                    tempspeedLimit = getCroppedImage(frame, x, y, radius)
                    if tempspeedLimit.shape[0]>0 and tempspeedLimit.shape[1]>0:
                        speedLimit=tempspeedLimit
                        error = new_error
                        cv2.circle(frame, (x,y), int(radius), (255,0,0))

    return True,speedLimit,edge,frame

def runOCR(im):
    #out = np.zeros(im.shape,np.uint8)
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)

    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    result = ""
    for cnt in reversed(contours):
        if cv2.contourArea(cnt)>50:
            [x,y,w,h] = cv2.boundingRect(cnt)
            if  h>28:
                cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
                roi = thresh[y:y+h,x:x+w]
                roismall = cv2.resize(roi,(10,10))
                roismall = roismall.reshape((1,100))
                roismall = np.float32(roismall)
                retval, results, neigh_resp, dists = model.find_nearest(roismall, k = 1)
                string = str(int((results[0][0])))
                #cv2.putText(out,string,(x,y+h),0,1,(0,255,0))
                result+=string
    #apply roll/pitch/yaw
    return result

cap = cv2.VideoCapture("./test4.mp4")
while not cap.isOpened():
    cap = cv2.VideoCapture("./test4.mp4")
    cv2.waitKey(1000)
    print "Wait for the header"

samples = np.loadtxt('generalsamples.data',np.float32)
responses = np.loadtxt('generalresponses.data',np.float32)
responses = responses.reshape((responses.size,1))

model = cv2.KNearest()
model.train(samples,responses)

pos_frame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
while True:
    flag, frame = cap.read()
    if flag:
        # The frame is ready and already captured
        pos_frame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
        print "Frame "+str(pos_frame)
        (val,crop,edges,fram) = hasSpeedLimitSign(frame)
        if val:
            new_speed = runOCR(crop)
            print new_speed
        cv2.imshow('Canny Edges', fram)
        #time.sleep(0.2)
    else:
        # The next frame is not ready, so we try to read it again
        cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, pos_frame-1)
        print "frame is not ready"
        # It is better to wait for a while for the next frame to be ready
        cv2.waitKey(1000)

    if cv2.waitKey(10) == 27:
        break
    if cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES) == cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT):
        # If the number of captured frames is equal to the total number of frames,
        # we stop
        break
