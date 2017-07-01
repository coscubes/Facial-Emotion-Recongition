import cv2
import os

images = os.listdir('jaffe/')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

for i in images:
	img = cv2.imread('jaffe/' + i, 0)
	faces = face_cascade.detectMultiScale(img, 1.3, 5)
	for (x, y, w, h) in faces:
		roi = img[y+5:y+h-5, x+5:x+w-5]
		temp = cv2.resize(roi,(160, 160), interpolation = cv2.INTER_CUBIC)
		cv2.imwrite("jaffefaces/" + i, temp)