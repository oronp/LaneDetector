import cv2
import numpy as np
import matplotlib.pyplot as plt

def make_coordinates(image,line_parameters):
    slope,intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1,y1,x2,y2])

def average_slope_intercept(image,line):
    left_fit = []
    right_fit = []
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        paramaters = np.polyfit((x1,x2),(y1,y2),1)
        slope = paramaters[0]
        intercept = paramaters[1]
        if slope < 0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))
    left_fit_average = np.average(left_fit,axis=0)
    right_fit_average = np.average(right_fit,axis=0)
    left_line = make_coordinates(image,left_fit_average)
    right_line = make_coordinates(image,right_fit_average)
    return np.array([left_line,right_line])


def Canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # change the chosen pic to chosen colors.
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def display_lines(image,lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
    return line_image

def regionOfInterest(image):
    height = image.shape[0] #takes the number of rows of the image matrix.
    polygons = np.array([
        [(200,height),(1100,height),(550,250)]
    ]) #create a matrix shape as a triangle
    mask = np.zeros_like(image) #create an exact same size pic as argument 1 in the color of black.
    cv2.fillPoly(mask,polygons,255) #fiill the argument 1 with argument 2 in the color of argument 3.
    masked_image = cv2.bitwise_and(image,mask)
    return masked_image

# image = cv2.imread("RoadPic.jpeg")  # reads a picture.
# lane_image = np.copy(image) #creating a copy of the original picture.
# canny_image = Canny(lane_image)
# cropped_image = regionOfInterest(canny_image)
# lines = cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
# #1. which image 2. height of bins 3.width of bins 4.array to work with 5. min to connect 6. max to connect
# averaged_lines = average_slope_intercept(lane_image,lines)
# line_image = display_lines(lane_image,averaged_lines)
# combo_image = cv2.addWeighted(lane_image, 0.8, line_image,1,1)
# cv2.imshow("POPUP",combo_image)  # write the picture -> get two arguments: 1. the name of the window will pop out. 2.the image variable.
# cv2.waitKey(0)  # wait command to keep the window up until the user click anything.
# #plt.imshow(canny)  # write the picture -> get one argument: the image variable.
# #plt.show() #shows the output without limited time.


cap = cv2.VideoCapture("test2.mp4")
while(cap.isOpened()):
    _, frame = cap.read()
    canny_image = Canny(frame)
    cropped_image = regionOfInterest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    # 1. which image 2. height of bins 3.width of bins 4.array to work with 5. min to connect 6. max to connect
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow("POPUP",
               combo_image)  # write the picture -> get two arguments: 1. the name of the window will pop out. 2.the image variable.
    if cv2.waitKey(1) == ord('a'):
        break
cap.release()
cv2.destroyAllWindows()