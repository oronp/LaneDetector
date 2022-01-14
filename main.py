import cv2
import numpy as np
import matplotlib.pyplot as plt

def Canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # change the chosen pic to chosen colors.
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def regionOfInterest(image):
    height = image.shape[0] #takes the number of rows of the image matrix.
    polygons = np.array([
        [(200,height),(1100,height),(550,250)]
    ]) #create a matrix shape as a triangle
    mask = np.zeros_like(image) #create an exact same size pic as argument 1 in the color of black.
    cv2.fillPoly(mask,polygons,255) #fiill the argument 1 with argument 2 in the color of argument 3.
    masked_image = cv2.bitwise_and(image,mask)
    return masked_image

image = cv2.imread("RoadPic.jpeg")  # reads a picture.
lane_image = np.copy(image) #creating a copy of the original picture.
canny = Canny(lane_image)
cropped_image = regionOfInterest(canny)
cv2.imshow("POPUP",cropped_image)  # write the picture -> get two arguments: 1. the name of the window will pop out. 2.the image variable.
cv2.waitKey(0)  # wait command to keep the window up until the user click anything.
#plt.imshow(canny)  # write the picture -> get one argument: the image variable.
#plt.show() #shows the output without limited time.
