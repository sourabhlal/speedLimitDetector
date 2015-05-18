import numpy as np
import matplotlib.pyplot as plt
import cv2


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

			_y1 = np.sqrt(r*r - (_x1-h)*(_x1-h)) + k
			_y2 = np.sqrt(r*r - (_x2-h)*(_x2-h)) + k
			#print "({},{}), ({},{})".format(_x1, _y1, _x2, _y2)
			dis1 = (cX - _x1) * (cX - _x1) + (cY - _y1) * (cY - _y1)
			dis2 = (cX - _x2) * (cX - _x2) + (cY - _y2) * (cY - _y2)

			if(dis1 > dis2):
				err += dis2
			else:
				err += dis1

	return float(err) / len(points)







def getEdges(img, height, width) :
	edges = cv2.Canny(img, height, width)
	return edges

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

im = cv2.imread('roadSign640x401.jpg')
imgray = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
edge = getEdges(imgray, 640, 401)
(rows, column) = edge.shape

draw = np.zeros(rows, dtype=np.int8)

t = cv2.findContours(edge, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
contours = [cv2.approxPolyDP(cnt, 3, True) for cnt in t[0]]



for i in range(0,len(contours)):
	if(len(contours[i]) > 5):
		center, radius = cv2.minEnclosingCircle(contours[i])
		if radius < im.shape[0]/8 :
			x, y = center
			x = int(x)
			y = int(y)
			error = 100000
			error = getDeviation(contours[i], x, y, radius)
			print  "center : ({},{})  radius: {}  error: {}".format(x, y, radius ,error)
			# cv2.circle(edge, (x,y), int(radius), (255,0,0))
			speedLimit = getCroppedImage(im, x, y, radius)
			cv2.circle(im, (x,y), int(radius), (255,0,0))


plt.subplot(122),plt.imshow(im, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(121),plt.imshow(edge, cmap = 'gray')
plt.title('Detected Image'), plt.xticks([]), plt.yticks([])

plt.show()