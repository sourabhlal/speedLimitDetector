import numpy as np
import matplotlib.pyplot as plt
import cv2


def getDeviation(points, h, k, r):
	err = 0
	# shift the center to zero
	x = 0
	y = 0
	for i in range(0, len(points)):
		_cX = points[i][0][0]
		_cY = points[i][0][1]
		#shift points
		cX = _cX - h
		cY = _cY - k

		# get the slope of line from center to the contour point
		m = (y - cY) / (x - cX)
		#y = mx + c
		cLine = y - m*x

		#get the corresponding point in circumference
		#  x^2 + m^2 * x^2 + 2 * m * x * cLine + cLine^2 = r^2
		# (1+m^2)*x^2 + 2*m*cLine*x + cLine^2 - r^2 = 0
		a = 1 + m * m
		b =  2 * m * cLine
		c = cLine * cLine - r * r

		det = b*b-4*a*c
		if det >= 0 :
			_x1 = (-b + np.sqrt(det) )/ 2*a
			_x2 = (-b - np.sqrt(det) )/ 2*a

			_y1 = np.sqrt(r*r - _x1*_x1)
			_y2 = np.sqrt(r*r - _x2*_x2)

			dis1 = (cX - _x1) * (cX - _x1) + (cY - _y1) * (cY - _y1)
			dis2 = (cX - _x2) * (cX - _x2) + (cY - _y2) * (cY - _y2)
			if(dis1 > dis2):
				err += dis2
			else:
				err += dis1

	return err / len(points)







def getEdges(img, height, width) :
	edges = cv2.Canny(img, height, width)
	return edges

im = cv2.imread('roadSign352x255.jpg')
imgray = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
edge = getEdges(imgray, 352, 255)
(rows, column) = edge.shape

draw = np.zeros(rows, dtype=np.int8)

t = cv2.findContours(edge, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
contours = [cv2.approxPolyDP(cnt, 3, True) for cnt in t[0]]



for i in range(0,len(contours)):
	if(len(contours[i]) > 5):
		center, radius = cv2.minEnclosingCircle(contours[i])
		if radius < im.shape[0]/8:
			x, y = center
			x = int(x)
			y = int(y)
			error = getDeviation(contours[i], x, y, radius)
			print  "radius: {}  error: {}".format(  x , y , radius ,error)
			cv2.circle(edge, (x,y), int(radius), (255,0,0))
			if error < 100:
				cv2.circle(im, (x,y), int(radius), (255,0,0))
		else:
			cv2.circle(im, (x,y), int(radius), (255,0,0))

plt.subplot(122),plt.imshow(im, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(121),plt.imshow(edge, cmap = 'gray')
plt.title('Detected Image'), plt.xticks([]), plt.yticks([])

plt.show()