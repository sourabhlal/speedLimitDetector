import cv2
import numpy as np
import time
from matplotlib import pyplot as plt

# Code to parse video frame by frame

#Bidesh is working on this
def cannyEdgeDetection(frame):
    return cv2.Canny(frame,150,250)

#based on paper
def hasSpeedLimitSign(frame):
    edges = cannyEdgeDetection(frame)
    return True,frame,edges

def runOCR(crop,frame):
    #run ocr on crop
    #apply roll/pitch/yaw
    #print result on frame
    return frame

cap = cv2.VideoCapture("./test5.mp4")
while not cap.isOpened():
    cap = cv2.VideoCapture("./test5.mp4")
    cv2.waitKey(1000)
    print "Wait for the header"

pos_frame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
while True:
    flag, frame = cap.read()
    if flag:
        # The frame is ready and already captured
        pos_frame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
        print "Frame "+str(pos_frame)
        (val,crop,edges) = hasSpeedLimitSign(frame)
        if val:
            frame = runOCR(crop,frame)
        cv2.imshow('Canny Edges', edges)
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


'''
img = cv2.imread('test.jpg',0)
edges = cv2.Canny(img,400,400)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()
'''