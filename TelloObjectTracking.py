from djitellopy import Tello
import cv2 as cv
import numpy as np


##
width = 640
height = 480
deadZone = 100
#######

startCounter = 0

#connect to drone
me = Tello()
me.connect()
me.for_back_velocity = 0
me.left_right_velocity = 0
me.up_down_velocity = 0
me.yaw_velocity = 0
me.speed = 0



print(me.get_battery())

me.streamoff()
me.streamon()
#########

frameWidth = width
frameHeight = height






global imgContour
global dir
def empty(a):
    pass

cv.namedWindow('HSV')
cv.resizeWindow('HSV', 640, 240)
cv.createTrackbar('HUE Min','HSV',19,179,empty)
cv.createTrackbar('HUE Max','HSV',35,179,empty)
cv.createTrackbar('SAT Min','HSV',107,255,empty)
cv.createTrackbar('SAT Max','HSV',255,255,empty)
cv.createTrackbar('VALUE Min','HSV',89,255,empty)
cv.createTrackbar('VALUE Max', 'HSV',255,255,empty)

cv.namedWindow('Parameters')
cv.resizeWindow('Parameters',640,240)
cv.createTrackbar('Threshold1', 'Parameters', 166,255,empty)
cv.createTrackbar('Threshold2', 'Parameters', 171, 255, empty)
cv.createTrackbar('Area', 'Parameters', 3750, 30000, empty)


def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0,cols):
                if imgArray[x][y].shape[:2] == imgArray[x][y].shape[:2]:
                    imgArray[x][y] = cv.resize(imgArray[x][y], (0,0), None, scale, scale)
                else:
                    imgArray[x][y] = cv.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale,scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv.cvtColor(imgArray[x][y], cv.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height,width,3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0,rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0,rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv.resize(imgArray[x], (0,0), None, scale, scale)
            else:
                imgArray[x] = cv.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv.cvtColor(imgArray[x], cv.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver

def getContours(img, imgContour):

    contours, heirarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv.contourArea(cnt)
        areaMin = cv.getTrackbarPos('Area', 'Parameters')
        if area > areaMin:

            cv.drawContours(imgContour, cnt, -1, (255,0,255), 7)
            peri = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, 0.02 * peri, True)
            #print(len(approx))
            x, y, w, h = cv.boundingRect(approx)
            #cv.rectangle(imgContour, (x,y), (x+w, y+h), (0,255, 0), 5)
        
            #cv.putText(imgContour, 'Points: ' + str(len(approx)), (x + w + 20, y + 20), cv.FONT_HERSHEY_COMPLEX, .7, (0,255,0), 2)

            #cv.putText(imgContour, 'Area: ' + str(int(area)), (x + w + 20, y + 45), cv.FONT_HERSHEY_COMPLEX, .7, (0,255,0), 2)

            #cv.putText(imgContour, ' ' + str(int(y)), (x - 20, y - 45), cv.FONT_HERSHEY_COMPLEX, .7, (0,255,0), 2)


            cx = int(x+(w/2))
            cy = int(y+(h/2))

            if (cx < int(frameWidth/2)-deadZone):
                cv.putText(imgContour, " GO LEFT ", (20, 50), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255),3)
                cv.rectangle(imgContour, (0, int(frameHeight/2 - deadZone)), (int(frameWidth/2)-deadZone, int(frameHeight/2)+deadZone), (0,0,255), cv.FILLED)
                dir = 1
            elif (cx > int(frameWidth/2) +deadZone):
                cv.putText(imgContour, " GO RIGHT ", (20, 50), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255),3)
                cv.rectangle(imgContour, (int(frameWidth/2 + deadZone), int(frameHeight/2 - deadZone)), (frameWidth, int(frameHeight/2)+deadZone), (0,0,255), cv.FILLED)
                dir = 2
            elif (cy< int(frameHeight/2) - deadZone):
                cv.putText(imgContour, " GO UP ", (20, 50), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255),3)
                cv.rectangle(imgContour, (int(frameWidth/2-deadZone), 0), (int(frameWidth/2 +deadZone), int(frameHeight/2)-deadZone), (0,0,255), cv.FILLED)
                dir = 3
            elif (cy > int(frameHeight / 2) + deadZone):
                cv.putText(imgContour, " GO DOWN ", (20, 50), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255),3)
                cv.rectangle(imgContour, (int(frameWidth/2 - deadZone), int(frameHeight/2) + deadZone), (int(frameWidth/2+deadZone), frameHeight), (0,0,255), cv.FILLED)
                dir = 4
            else: dir = 0


            cv.line(imgContour, (int(frameWidth/2), int(frameHeight/2)), (cx,cy), (0,0,255), 3)


def display(img):
    cv.line(img, (int(frameWidth/2)-deadZone,0), (int(frameWidth/2)-deadZone,frameHeight), (255,255,0),3)
    cv.line(img, (int(frameWidth/2)+deadZone,0), (int(frameWidth/2)+deadZone,frameHeight), (255,255,0),3)

    cv.circle(img, (int(frameWidth/2), int(frameHeight/2)), 5, (0,0,255), 5)
    cv.line(img, (0, int(frameHeight/2) - deadZone), (frameWidth, int(frameHeight/2) - deadZone), (255, 255, 0), 3)
    cv.line(img, (0, int(frameHeight/2) + deadZone), (frameWidth, int(frameHeight/2) + deadZone), (255, 255, 0), 3)

while True:

    frame_read = me.get_frame_read()
    myFrame = frame_read.frame
    img = cv.resize(myFrame, (width, height))
    imgContour = img.copy()
    imgHsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    h_min = cv.getTrackbarPos('HUE Min', 'HSV')
    h_max = cv.getTrackbarPos('HUE Max', 'HSV')
    s_min = cv.getTrackbarPos('SAT Min', 'HSV')
    s_max = cv.getTrackbarPos('SAT Max', 'HSV')
    v_min = cv.getTrackbarPos('VALUE Min', 'HSV')
    v_max = cv.getTrackbarPos('VALUE Max', 'HSV')
    print(h_min)

    lower = np.array([h_min, s_min,v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv.inRange(imgHsv, lower, upper)
    result = cv.bitwise_and(img, img, mask = mask)
    mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)

    imgBlur = cv.GaussianBlur(result, (7,7), 1)
    imgGray = cv.cvtColor(imgBlur, cv.COLOR_BGR2GRAY)
    threshold1 = cv.getTrackbarPos('Threshold1', 'Parameters')
    threshold2 = cv.getTrackbarPos('Threshold2', 'Parameters')
    imgCanny = cv.Canny(imgGray, threshold1, threshold2)
    kernel = np.ones((5,5))
    imgDil = cv.dilate(imgCanny, kernel, iterations=1)
    getContours(imgDil, imgContour)
    display(imgContour)

 

    if startCounter == 0:
        me.takeoff()
        startCounter = 1

    if dir == 1:
        me.yaw_velocity = -60
    elif dir == 2:
        me.yaw_velocity = 60
    elif dir == 3:
        me.up_down_velocity = 20
    elif dir == 4:
        me.up_down_velocity = -20
    else:
        me.left_right_velocity =0; me.for_back_velocity = 0; me.up_down_velocity = 0; me.yaw_velocity=0;
    # send velocity values to tello drone
    if me.send_rc_control:
        me.send_rc_control(me.left_right_velocity, me.for_back_velocity, me.up_down_velocity, me.yaw_velocity)
    print(dir)
    stack = stackImages(0.9, ([img,result], [imgDil, imgContour]))

    cv.imshow('Horizontal Stacking', stack)


    if cv.waitKey(1) & 0xFF == ord('q'):
        me.land()
        break