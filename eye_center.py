import numpy as np
from math import sqrt
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

frame = cv2.imread("t3.jpg")

#img has to be grayscale
def eye_center(img,grad_x,grad_y):
	center_x=0
	center_y=0
	max_c=0
	for j in range(len(img)):
		for i in range(len(img[0])):
			c=c_compute(img,j,i,grad_x,grad_y)
			if c>max_c:
				max_c=c
				center_y=j
				center_x=i
			else:
				continue
	print center_y
	print center_x
	return [center_y,center_x]

def c_compute(img,point_y,point_x,grad_x,grad_y):
	d=0
	for n in range(len(grad_y)):
		for m in range(len(grad_x[0])):
			dx=m-point_x
			dy=n-point_y
			if dx==0 and dy==0:
				continue
			magnitude=sqrt(dx**2+dy**2)
			dx=dx/magnitude
			dy=dy/magnitude
			grad_mag = sqrt((grad_x[n][m])**2+(grad_y[n][m])**2)
			d=d+((dx*(grad_x[n][m]))+(dy*(grad_y[n][m])))*(-1)
			#print(((dx*grad_x[n][m])+(dy*grad_y[n][m])))
	#print point_y
	#print point_x
	print d
	return d

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray)
for (x,y,w,h) in faces:
    cv2.rectangle(frame,(x,y),(x+h,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h,x:x+w]
    roi_color = frame[y:y+h,x:x+h]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
    	cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    	roi_eye=cv2.GaussianBlur(roi_gray[ey:ey+eh,ex:ex+ew],(3,3),0)
    	new_ones=np.ones((len(roi_eye),len(roi_eye[0])))
    	roi_eye1=new_ones-roi_eye
    	grad=np.gradient(roi_eye1)
    	grad_x=grad[1]
    	grad_y=grad[0]
    	d=eye_center(roi_eye1,grad_x,grad_y)
    	cv2.circle(roi_color,(d[1]+ex,d[0]+ey),2,(255,0,0),4)
#img has to be a grayscale


cv2.imshow("roi_color",roi_color)
cv2.waitKey(0)
cv2.destroyAllWindows()