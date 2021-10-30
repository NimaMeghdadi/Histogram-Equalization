import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
from PIL import Image
import itertools

def drow_Result(imgAddr, title, location, axis):
    fig.add_subplot(rows, columns, location)
    plt.imshow(imgAddr,cmap='gray')
    plt.axis(axis)
    plt.title(title)

#imagePath = './testGray.jpg'
#imagePath = './testColorful.jpg'
imagePath = './testColorful2.jpg'
#imagePath = './testColorful3.jpg'

beforeImage = Image.open(imagePath)

image = Image.open(imagePath)
image = image.convert("L")  # convert to gray scale

width, height = image.size
totalPixels = width * height

cProbability = [0] * 256  

freq = image.histogram()
a = np.array(image)
plt.hist(a.ravel(),  bins=256) #Diagram
plt.ylabel('Probability')
plt.xlabel('Gray Level')
#plt.savefig('inputhist.svg')

prevSum = 0
for i in range(256):
    prevSum += freq[i]*1.0/totalPixels  
    cProbability[i] = prevSum

print(cProbability[255]) #show Probability
images = np.array(Image.open(imagePath))
b, g, r = cv2.split(images)

if cProbability[255] > 0.005:
    R, G, B = images[:, :, 0], images[:, :, 1], images[:, :, 2]
    images = 0.2989 * R + 0.5870 * G + 0.1140 * B

pixels = image.load()  
for x, y in itertools.product(range(width), range(height)):
    pixels[x, y] = int((255 * cProbability[pixels[x, y]]))

a = np.array(image)
plt.hist(a.ravel(),  bins=256)

fig = plt.figure(figsize=(10, 7))
rows = 2
columns = 2

drow_Result(beforeImage, "Before", 1, "off")
imageGray = Image.open(imagePath).convert('LA')
drow_Result(imageGray, "Gray", 2, "off")
drow_Result(image, "After", 3, "off")
plt.show()