import pandas as pd
import numpy as np
import cv2

df = pd.read_csv("fer2013/fer2013.csv")

counter1, counter2, counter3 = 0, 0, 0

for i, rows in df.iterrows():
	print i
	image = rows['pixels']
	image = np.array(image.split(), np.uint8)
	image = np.reshape(image, (48, 48))

	if rows['Usage'] == "Training":
		counter1 += 1
		cv2.imwrite("training/" + str(rows['emotion']) + "_" + str(counter1) + ".jpg", image) 

	elif rows['Usage'] == "PublicTest":
		counter2 += 1
		cv2.imwrite("publictest/" + str(rows['emotion']) + "_" + str(counter2) + ".jpg", image) 

	elif rows['Usage'] == "PrivateTest":
		counter3 += 1
		cv2.imwrite("privatetest/" + str(rows['emotion']) + "_" + str(counter3) + ".jpg", image) 

print counter1, counter2, counter3