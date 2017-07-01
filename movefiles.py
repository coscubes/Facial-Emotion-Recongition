import os
from shutil import copyfile

folders = os.listdir('KDEF/')

for i in folders:
	temp = os.listdir('KDEF/' + i)
	for file in temp:
		if file.endswith(".JPG"):
			copyfile("KDEF/" + i + "/" + file, "KDEF_all/" + file)
