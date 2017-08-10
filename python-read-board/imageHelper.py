import cv2
#from pylab import *
#import pylab as pl
#import matplotlib.pyplot as plt
import numpy as np

from helpers import getRect, isPointInContour

def convertToGray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def convertToColor(image):
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

def drawLine(image, start, end, color, colorWidth):
    cv2.line(image, (start[0], start[1]), (end[0], end[1]), color, colorWidth)
    return

def putText(image, point, text):
    cv2.putText(image, text, point, cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 5)
    return

def show(image, title=None):
    #   show image   #
    #_=pl.axis("off")
    #if (title != None):
    #    _=pl.title(title)
    #_ = pl.imshow(image, cmap=pl.gray())
    #plt.show()
    return

def getColorProps2(image):
    image1 = image.flatten()
    pixels = [image1[i] for i in range(image1.shape[0]) if image1[i] != 255]
    avg = np.average(pixels, axis=0)
    min = np.amin(pixels, axis=0)
    max = np.amax(pixels, axis=0)

    ret, thresh = cv2.threshold(image, avg, 255, cv2.THRESH_BINARY)
    image = 255 - thresh
    #show(image)

    return (min, max, avg)

def getColorProps(image, contour):
    #mask = np.zeros(image.shape, np.uint8)
    #cv2.drawContours(mask, [square], 0, 255, -1)
    #mean = cv2.mean(squareImage, mask=mask)
    border = 5
    x, y, w, h = getRect(contour)
    image = image[y + border:y + h - border,
                  x + border:x + w - border]
    average_color_per_row = np.average(image, axis=0)
    avg = np.average(average_color_per_row, axis=0)

    min_per_row = np.amin(image, axis=0)
    min =np.amin(min_per_row, axis=0)

    max_per_row = np.amax(image, axis=0)
    max = np.amax(max_per_row, axis=0)
    return (min, max, avg)

def putDigitInCenter(image):
    height, width = image.shape[0], image.shape[1]
    boundSpace = 3
    smallAxis, bigAxis = min([height, width]), max([height, width])
    diff = bigAxis - smallAxis
    if (height > width):
        minX, maxX = boundSpace + int(diff / 2), boundSpace + int(diff / 2) + smallAxis
        minY, maxY = boundSpace, boundSpace + bigAxis
    else:
        minX, maxX = boundSpace, boundSpace + bigAxis
        minY, maxY = boundSpace + int(diff / 2), boundSpace + int(diff / 2) + smallAxis

    biggerImage = np.zeros(((boundSpace * 2) + bigAxis, (boundSpace * 2) + bigAxis), dtype=np.uint8)
    biggerImage[minY:maxY, minX:maxX] = image

    return biggerImage

    # Thresholding
    #_, biggerImage = cv2.threshold(biggerImage,127,255,cv2.THRESH_TOZERO)
    # eroding the image
    #kernel = np.ones((5, 5), np.uint8)
    #biggerImage = cv2.erode(biggerImage, kernel, iterations=1)

def percentageOfWhitePixels(image):
    whitePixels = cv2.countNonZero(image)
    percentOfWhite = int((whitePixels / (image.shape[0] * image.shape[1])) * 100)
    return percentOfWhite