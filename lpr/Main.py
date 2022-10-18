import glob
import cv2
import numpy as np
import os
import time
import datetime
import csv

from lpr import DetectChars
from lpr import DetectPlates
from lpr import PossiblePlate

SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

def recognize(image,saveChars,saveDay,showBox):
    CnnClassifier = DetectChars.loadCNNClassifier()
    if CnnClassifier == False:
        print("\nerror: CNN traning was not successful\n")
        return 
    
    imgOriginalScene = image.copy()
    image1 = image.copy()
    if imgOriginalScene is None:                            
        print("\nerror: image not read from file \n\n")      
        os.system("pause")                                  
        return                                              

    listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)           
                                                                                        
    listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)        

    plateChars = ''
    codeDay    = ''
    numChars   = ''
    violate    = ''
    ret        = 9
    listOfPossiblePlates.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)
    black = np.zeros((480,640,3),np.uint8)
    
    if len(listOfPossiblePlates) == 0:
        ret = 0
        image1 = imgOriginalScene
        print('No Plate')
    else:
        licPlate = listOfPossiblePlates[0]                                                   
        plateChars = listOfPossiblePlates[0].strChars
        numChars = len(plateChars)
        if numChars > 0: 
            displayMsg,codeDay,showBox,ret = checkDay(plateChars,
            codeDay,
            saveChars,
            saveDay,
            showBox,
            ret)
            if showBox == 'Y':
                dayToday = datetime.date.today().strftime("%A")
                if dayToday == codeDay:
                    colorBox = SCALAR_RED
                    violate  = 'Y'
                else:
                    colorBox = SCALAR_GREEN
                    violate  = 'N'
                image1,black,ret = drawBox(image,
                black,
                licPlate,
                displayMsg,
                colorBox,
                ret)
    return image1, black, plateChars, codeDay, violate, showBox, ret

def checkDay(plateChars,codeDay,saveChars,saveDay,showBox,ret):
    validChars = ''
    lastNum    = ''
    if len(plateChars) in [6,7]:
        showBox = 'Y'
        validChars = plateChars
        lastNum = plateChars[-1]
        lastNum = int(lastNum)
        if  lastNum in [1,2]:
            codeDay = 'Monday'
            ret = 1
        elif lastNum in [3,4]:
            codeDay = 'Tuesday'
            ret = 2
        elif lastNum in [5,6]:
            codeDay = 'Wednesday'
            ret = 3
        elif lastNum in [7,8]:
            codeDay = 'Thursday'
            ret = 4
        elif lastNum in [9,0]:
            codeDay = 'Friday'
            ret = 5
        print('Good Plate')
    else:
        validChars = saveChars
        codeDay = saveDay
        ret = 6
        print('Not 6-7 ' + str(len(plateChars)))
    displayMsg = ' ' + validChars + ' (Coding: ' + codeDay + ')'
    return displayMsg, codeDay, showBox, ret
 
def drawBox(image, black, licPlate, displayMsg, colorBox, ret):
    imgcrop = np.zeros((210,585,3),np.uint8)
    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)            
    xmin = int(p2fRectPoints[1][0])
    ymin = int(p2fRectPoints[1][1])
    xmax = int(p2fRectPoints[3][0])
    ymax = int(p2fRectPoints[3][1])
    dims = [xmin,ymin,xmax,ymax]
    checkNeg = any(n<0 for n in dims)
    if checkNeg == True:
        ret = 0
        print('maybe wrong')
        return image, black, ret
    else:
        width  = xmax - xmin
        height = ymax - ymin
        area   = width * height
        print(width,height,area)

        if height < 50 and area <= 10000:
            ret = 9
            print('maybe wrong')
            return image, black, ret

        else:
            pts = np.array([[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]], np.int32)
            pts = pts.reshape((-1,1,2))
            cv2.polylines(image,[pts],True,colorBox,2)    

            widthA  = xmax - xmin
            fontScale = widthA / 1000
            if fontScale > 0.5:
                add1  = 25
                add2  = 35
                scale = 0.7
            else:
                add1  = 15
                add2  = 25
                scale = 0.5

            yLefta  = ymax + add1
            yLeft1  = ymax + add2
            yRight1 = ymax + add2

            pts = np.array([[xmin,ymax],[xmax,ymax],[xmax,yRight1],[xmin,yLeft1]],np.int32)
            pts = pts.reshape((-1,1,2))
            cv2.fillPoly(image,[pts],colorBox,8)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image,displayMsg,(xmin,yLefta), font, scale,(0,0,0),1,cv2.LINE_AA)
        
            plateImg = image[ymin:yLeft1,xmin:xmax]
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv[...,2] = hsv[...,2]*0.2 # 0.1-0.9, 0.1 darkest
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            image[ymin:yLeft1,xmin:xmax] = plateImg

            plateCopy  = plateImg.copy()
            imgcrop = cv2.resize(plateCopy,(585,210))
            black[135:345,28:613] = imgcrop
    
    return image, black, ret