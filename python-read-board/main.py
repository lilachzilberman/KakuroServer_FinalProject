# -*- coding: utf-8 -*-
import cv2
import numpy as np
import math
import json
import sys

from mnist.main import run as getDigitsFromMNIST
from mnist.config import size as sizeToMNIST
from helpers import getAllContours, getContourCenter, isCenterEqual, isPointInContour, calcDistance, cut_out_sudoku_puzzle, thresholdify, dilate, new_dilate, getAllTriangles, getAllSquares
from helpers import getContourApprox, getMiddleVertex, getRightVertex, getLeftVertex, getTopLeft, getBottomRight, findContourAndRectOfPoint, getRect, threshPost, invertProcess, postForTriangles, threshPostAllSquares
from helpers import containsAnyContour, containedByOtherContour, threshForSquares, postForBlocked, threshForBlock

from imageHelper import show, drawLine, convertToGray, convertToColor, putDigitInCenter, getColorProps, percentageOfWhitePixels, getColorProps2
SAFETY_PIXEL_WIDTH = 3

def preProcess(image):
    #show(image)
    #TODO: delete this line
    image = convertToGray(image)
    #show(image)

    #image = dilate(image)
    #show(image)

    image = thresholdify(image)
    #show(image)
    #show(image)
    image = cv2.GaussianBlur(image, (7, 7), 0)
    #show(image)
    #show(image)
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)) #TODO: was 2,2
    #image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    #image = dilate(thresh)
    #show(image)
    return image

def postProcess(image):
    image = threshPost(image)
    # dilating
    #image = new_dilate(image, 3) #TODO: was 7
    #show(image)
    #image = thresholdify(image)
    #show(image)
    # eroding
    #kernel = np.ones((3, 3), np.uint8)
    #image = cv2.erode(image, kernel, iterations=1)
    #show(image)
    # blurring
    #image = cv2.GaussianBlur(image, (5, 5), 15)
    #show(image)
    return image

def crop(image):
    contours = getAllSquares(getAllContours(image.copy()))
    contour = max(contours, key=cv2.contourArea)
    board = cut_out_sudoku_puzzle(image.copy(), contour)
    rect = getRect(contour)
    return board, rect

def getBoardFromImage(orig):
    image = preProcess(orig)
    cropedImage, rect = crop(image)
    return cropedImage, rect

def checkIfFarBiggerThanAreaSize(size, contour):
    # If the contour is bigger than 50% of the board
    return (cv2.contourArea(contour) > (size / 2))

def areaBiggerThan(size, contour):
    # If the contour is bigger than 50% of the board
    return (cv2.contourArea(contour) > size)

def checkIfVeryBelowAreaSize(areaSize, contour):
    # If the contour is smaller than 60% of the average size
    return (cv2.contourArea(contour) < (areaSize * 0.6))

def checkIfBlockingCell(maxPixelValue):
    return maxPixelValue > 151

def getTwinContour(source, contours):
    sourceCenter = getContourCenter(source)
    contoursWithoutSource = list(filter(lambda x: not isCenterEqual(sourceCenter, x), contours))
    minDist = 10000000 #default value
    # getting the closest contour to sourceCenter
    closest = None
    for contour in contoursWithoutSource:
        currCenter = getContourCenter(contour)
        currDist = calcDistance(sourceCenter, currCenter)
        if (currDist < minDist):
            minDist = currDist
            closest = contour
    return closest

def drawSquare(image, topLeft, topRight, bottomRight, bottomLeft):
    colorOut = (255, 255, 255) # white
    colorIn = (0, 0, 0) # black
    outWidth = SAFETY_PIXEL_WIDTH
    inWidth = SAFETY_PIXEL_WIDTH

    image = convertToColor(image)

    # Drawing the outer border
    # top right to bottom right
    drawLine(image, topRight, bottomRight, colorOut, outWidth)
    # bottom right to bottom left
    drawLine(image, bottomRight, bottomLeft, colorOut, outWidth)
    # bottom left to top left
    drawLine(image, bottomLeft, topLeft, colorOut, outWidth)
    # top left to top right
    drawLine(image, topLeft, topRight, colorOut, outWidth)

    # Drawing the inner border
    # top right to bottom right
    drawLine(image, (topRight[0]-outWidth, topRight[1]+outWidth),
                    (bottomRight[0]-outWidth, bottomRight[1]-outWidth),
             colorIn, inWidth)
    # bottom right to bottom left
    drawLine(image, (bottomRight[0]-outWidth, bottomRight[1]-outWidth),
                    (bottomLeft[0]+outWidth, bottomLeft[1]-outWidth),
             colorIn, inWidth)
    # bottom left to top left
    drawLine(image, (bottomLeft[0]+outWidth, bottomLeft[1]-outWidth),
                    (topLeft[0]+outWidth, topLeft[1]+outWidth),
             colorIn, inWidth)
    # top left to top right
    drawLine(image, (topLeft[0]+outWidth, topLeft[1]+outWidth),
                    (topRight[0]-outWidth, topRight[1]+outWidth),
             colorIn, inWidth)

    image = convertToGray(image)
    return image

def drawSquaresOnTriangleCells(image, triangle_contours):
    # Converting the triangle contours into squares
    for cnt in triangle_contours:
        # todo: delete
        if (False):
            simage = convertToColor(image)
            simage = cv2.drawContours(simage, [cnt], -1, (0, 255, 0), 5)
            show(simage)

        # curr vars
        cX, cY = getContourCenter(cnt)
        approx = getContourApprox(cnt)
        middleVertex, isUpper = getMiddleVertex(approx, (cX, cY))

        # twin
        twin = getTwinContour(cnt, triangle_contours)
        twinCenterX, twinCenterY = getContourCenter(twin)
        twinApprox = getContourApprox(twin)
        twinMiddleVertex, twinIsUpper = getMiddleVertex(twinApprox, (twinCenterX, twinCenterY))

        # upper
        if (isUpper and not twinIsUpper):
            rightVertex = getRightVertex(approx, middleVertex)
            leftVertex = getLeftVertex(approx, middleVertex)

            twinRightVertex = getRightVertex(twinApprox, twinMiddleVertex)
            twinLeftVertex = getLeftVertex(twinApprox, twinMiddleVertex)

            topLeft = getTopLeft(leftVertex, twinLeftVertex)
            bottomRight = getBottomRight(rightVertex, twinRightVertex)
            # Drawing a square
            image = drawSquare(image, topLeft, middleVertex[0], bottomRight, twinMiddleVertex[0])

            # cv2.circle(image, (topLeft[0], topLeft[1]), 20, (255, 0, 0), -1)
            # cv2.circle(image, (bottomRight[0], bottomRight[1]), 20, (0, 0, 255), -1)
            # lower
            # else:
            # cv2.circle(image, (middleVertex[0][0], middleVertex[0][1]), 20, (255, 255, 0), -1)
            # Handling square cells

    boardSize = image.shape[0] * image.shape[1]
    # getting all square contours
    square_contours = getAllSquares(getAllContours(image))
    square_contours = list(filter(lambda x: containsAnyContour(x, triangle_contours), square_contours))
    # filter the board contour if exists
    square_contours = list(filter(lambda x: not checkIfFarBiggerThanAreaSize(boardSize, x), square_contours))
    # filter contours very below the average (noise contour)
    contourAvgSize = sum(cv2.contourArea(item) for item in square_contours) / float(len(square_contours))
    square_contours = list(filter(lambda x: not checkIfVeryBelowAreaSize(contourAvgSize, x), square_contours))
    return square_contours

def convertSemiCellsToCells(image):
    image = convertToGray(image)
    #show(image)
    image = postForTriangles(image)
    #show(image)
    #TODO: no converting to color
    if (False):
        stam1 = getAllSquares(getAllContours(image))
        stam = convertToColor(image)
        stam = cv2.drawContours(stam, stam1, -1, (0, 255, 0), 5)
        show(stam)

    # getting all triangle contours
    triangle_contours = getAllTriangles(getAllContours(image))

    if (False):
        stam = convertToColor(image)
        stam = cv2.drawContours(stam, triangle_contours, -1, (0, 255, 0), 5)
        show(stam)

    if (len(triangle_contours) == 0):
        return image, triangle_contours
    # filter contours very below the average (noise contour)
    contourAvgSize = sum(cv2.contourArea(item) for item in triangle_contours) / float(len(triangle_contours))
    triangle_contours = list(filter(lambda x: not checkIfVeryBelowAreaSize(contourAvgSize, x), triangle_contours))

    if (False):
        simage = convertToColor(image)
        simage = cv2.drawContours(simage, triangle_contours, -1, (0, 255, 0), 5)
        show(simage)

    onlyTriangleSquares = drawSquaresOnTriangleCells(image, triangle_contours)
    return onlyTriangleSquares, triangle_contours

def getBoardGrid(boardSize, cnts):
    # initialize the reverse flag and sort index
    reverse = False
    # sorting against the y-coordinate
    i = 1
    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [getRect(c) for c in cnts]

    # calculating the first square (the closest to (0,0) which is the upper left square)
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: calcDistance((0,0), (b[1][0], b[1][1])),
                                        reverse=reverse))
    leftCellInLine = (cnts[0], boundingBoxes[0])

    gridCells = []
    for line in range(1, boardSize + 1):
        lineCells = []
        lineCells.append(leftCellInLine)

        currCell = leftCellInLine
        for column in range(2, boardSize + 1):
            # current cell center
            (cX, cY) = getContourCenter(currCell[0])
            cellWidth = currCell[1][2]
            nextCellPoint = (cX + cellWidth, cY)
            nextCell = findContourAndRectOfPoint(nextCellPoint, zip(cnts, boundingBoxes))
            if nextCell == None:
                #print("can't find the next cell")
                return None
            lineCells.append(nextCell)
            currCell = nextCell

        gridCells.append(lineCells)

        # finding next line left cell
        if (line < boardSize):
            (cX, cY) = getContourCenter(leftCellInLine[0])
            cellHeight = leftCellInLine[1][3]
            nextCellPoint = (cX, cY + cellHeight)
            leftCellInLine = findContourAndRectOfPoint(nextCellPoint, zip(cnts, boundingBoxes))
            if leftCellInLine == None:
                #print("can't find the next line left cell")
                return None
    return gridCells

def handleSquareImage(origCroped, image):
    #TODO: use imageRect
    #TODO: just send to mnist or check the contour and then mnist
    # TODO: might be more than one digit?
    # counting the percentage of white pixels
    percentOfWhite = percentageOfWhitePixels(image)
    #show(image)
    #show(origImage)

    if percentOfWhite < 10:
        return { 'hasValue': True, 'data': None }
    else:
        return { 'hasValue': False, 'data': [image] }

alonW = []
alonH = []

def handleTriangleImage(origCroped, image, contour, minX, minY, alon):
    origGray = convertToGray(origCroped)
    if (True):
        # since we draw a square outside the triangle, we need to look for it's inner contours
        kernel = np.ones((3, 3), np.uint8)
        #image = cv2.erode(image, kernel, iterations=3)
        #image = cv2.GaussianBlur(image, (3, 3), 0)
        #show(image)
        stam = thresholdify(convertToGray(origCroped))
        stam = cv2.GaussianBlur(stam, (3, 3), 0)
        if (alon[0] == 2 and alon[1] == 0):
            a = 5
            #show(stam)

    digitContours = getAllContours(stam)
    # excluding all lines and other contours which are not cell square
    digitContours = list(filter(lambda x: not containedByOtherContour(x, digitContours), digitContours))

    digits = []
    for digitContour in digitContours:
        (x, y, w, h) = rect = getRect(digitContour)

        digitHeightInPercent, digitWidthInPercent = h / image.shape[0], w / image.shape[1]
        # not the crossing line of the triangle
        if ((digitWidthInPercent > 0.10 and digitWidthInPercent < 0.4) and
            (digitHeightInPercent > 0.10 and digitHeightInPercent < 0.7) and
            (x > 5 and y > 5)):
            # todo: debug
            if (alon[0] == 2 and alon[1] == 0):
                stam1 = convertToColor(stam)
                cv2.drawContours(stam1, [digitContour], -1, (255, 0, 0), 5)
                #show(stam1)
            digitCenter = getContourCenter(digitContour)
            # since we croped, we want to test the original image X,Y of the contour
            origDigitCenter = (digitCenter[0] + minX, digitCenter[1] + minY)
            # TODO: delete these 4 lines
            # cv2.drawContours(croped, [digitContour], -1, (0, 0, 0), 5)
            # show(croped)

            if (isPointInContour(origDigitCenter, contour)):
                global alonW
                global alonH
                alonW.append(digitWidthInPercent)
                alonH.append(digitHeightInPercent)
                digits.append({'contour': digitContour, 'rect': rect})

    # sorting the digits from the left to the right (x axis)
    digits = sorted(digits, key=lambda x: x['rect'][0])
    # todo: delete imageRect references
    #(imageX, imageY, w, h) = imageRect
    safeBorder = 3
    digitsWithBorder = []
    for digit in digits:
        (x, y, w, h) = digit['rect']
        digitImage = image[y - safeBorder:y + h + safeBorder,
                     x - safeBorder:x + w + safeBorder]

        if (False):
            p = cv2.GaussianBlur(digitImage, (7, 7), 0)
            thresh = cv2.adaptiveThreshold(p.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                           cv2.THRESH_BINARY, 11, 10)
                                            # 3 #TODO: was 11,10 or 11,7 or 5,2
            #show(digitImage)
            #p = putDigitInCenter(digitImage)
            #show(p)
            #value = getDigitsFromMNIST([p])
            #show(p, str(value[0]))
        # show(digitImage)
        # since we croped the digit from the croped image (minY)
        # since we croped the board from the original image (imageY). same goes for X
        # digitImage = origImage[y + minY + imageY - safeBorder: y + h + minY + imageY + safeBorder,
        #                       x + minX + imageX - safeBorder: x + w + minX + imageX + safeBorder]
        # show(255 - digitImage)
        # digitImage = convertToGray(255 - digitImage)

        digitImage = putDigitInCenter(digitImage)
        digitImage = cv2.resize(digitImage, (sizeToMNIST, sizeToMNIST))
        #show(digitImage)
        #thresh = cv2.adaptiveThreshold(digitImage.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       #cv2.THRESH_BINARY, 3, 10)
        #digitImage = cv2.GaussianBlur(digitImage, (7, 7), 0)
        #digitImage = cv2.blur(digitImage, (3, 3))
        digitImage = cv2.bilateralFilter(digitImage, 17, 75, 75)
        #show(digitImage)
        #show (thresh)
        #show(digitImage)
        digitsWithBorder.append(digitImage)

    if (len(digitsWithBorder) == 0):
        return { 'hasValue': True, 'data': None }
    else:
        return { 'hasValue': False, 'data': digitsWithBorder }

def handleDigitsFromImage(origImage, image, shapeContour, isSquare, alon):
    x, y = [], []
    for contour_lines in shapeContour:
        for line in contour_lines:
            x.append(line[0])
            y.append(line[1])
    minX, minY, maxX, maxY = min(x), min(y), max(x), max(y)

    croped = image[minY:maxY, minX:maxX]
    origCroped = origImage[minY:maxY, minX:maxX]

    if (isSquare):
        return handleSquareImage(origCroped, croped)
    else:
        return handleTriangleImage(origCroped, croped, shapeContour, minX, minY, alon)

def readCellFromImage(origImage, image, cell, allCells, alon):
    (regularCells, blockCells, triangles) = allCells
    (contour, rect) = cell

    if (False):
        simage = cv2.drawContours(origImage, [contour], -1, (0, 255, 0), 5)
        show(simage)

    trianglesInCell = []
    for triangle in triangles:
        if (False):
            simage = cv2.drawContours(origImage, [triangle], -1, (0, 255, 0), 5)
            show(simage)

        triangleCenter = getContourCenter(triangle)
        if (isPointInContour(triangleCenter, contour)):
            trianglesInCell.append({ 'contour': triangle, 'center': triangleCenter })

    # native square, no triangles
    if (len(trianglesInCell) == 0):

        isBlockCell = False
        for blockCell in blockCells:
            blockCenter = getContourCenter(blockCell)
            if (isPointInContour(blockCenter, contour)):
                isBlockCell = True
                break

        if (isBlockCell):
            return {
                'valid': True,
                'block': True
            }
        else:
            value = handleDigitsFromImage(origImage, image, contour, True, alon)
            return {
                'valid': True,
                'cell': { 'cellType': 'square', 'value': value }
            }

    elif (len(trianglesInCell) == 2):
        center1, center2 = trianglesInCell[0]['center'], trianglesInCell[1]['center']
        # if the triangles are not bottom left and upper right
        if ((center1[0] < center2[0] and center1[1] < center2[1]) or
            (center2[0] < center1[0] and center2[1] < center1[1])):
            return { 'valid': False }
        else:
            if (center1[0] < center2[0] and center1[1] > center2[1]):
                bottomLeftTriangle = trianglesInCell[0]['contour']
                upperRightTriangle = trianglesInCell[1]['contour']
            else:
                bottomLeftTriangle = trianglesInCell[1]['contour']
                upperRightTriangle = trianglesInCell[0]['contour']
            bottomValue = handleDigitsFromImage(origImage, image, bottomLeftTriangle, False, alon)
            upperValue = handleDigitsFromImage(origImage, image, upperRightTriangle, False, alon)
            return {
                'valid': True,
                'cell': {
                    'cellType': 'triangle',
                    'value': {
                        'hasValue': bottomValue['hasValue'] and upperValue['hasValue'],
                        'bottom': bottomValue,
                        'upper': upperValue
                    }
                }
            }

    else:
        return { 'valid': False }

def getDigitsFromImages(mnistCells):
    images, result = [], []

    for (i, j, cell) in mnistCells:

        if (cell['cellType'] == 'square'):
            digits = cell['value']
            result.append({ 'row': i, 'col': j, 'type': 'square', 'value': None })
            index = len(result) - 1
            for k in range(0, len(digits)):
                images.append({ 'image': digits[k], 'index': index })

        elif (cell['cellType'] == 'triangle'):
            bottom = cell['value']['bottom']
            if (bottom['hasValue'] == True):
                result.append({ 'row': i, 'col': j, 'type': 'bottom', 'value': bottom['data'] })
            else:
                digits = bottom['data']
                result.append({ 'row': i, 'col': j, 'type': 'bottom', 'value': None })
                index = len(result) - 1
                for k in range(0, len(digits)):
                    images.append({ 'image': digits[k], 'index': index })

            upper = cell['value']['upper']
            if (upper['hasValue'] == True):
                result.append({ 'row': i, 'col': j, 'type': 'upper', 'value': upper['data'] })
            else:
                digits = upper['data']
                result.append({ 'row': i, 'col': j, 'type': 'upper', 'value': None })
                index = len(result) - 1
                for k in range(0, len(digits)):
                    images.append({ 'image' : digits[k], 'index': index })

    #todo: make if faster!
    onlyDigits = []
    for item in images:
        onlyDigits.append(item['image'])

    digitsWithValues = getDigitsFromMNIST(onlyDigits)
    for i in range(0, len(images)):
        item = images[i]
        if (True):
            u = result[item['index']]
            str1 = "row:" + str(u['row']) + " col:" + str(u['col']) + " t:" + str(u['type']) + " val:" + str(digitsWithValues[i])
            #show(item['image'], str1)
        if (result[item['index']]['value'] == None):
            result[item['index']]['value'] = digitsWithValues[i]
        else:
            # since we sorted the cell's digits from the left to the right
            result[item['index']]['value'] *= 10
            result[item['index']]['value'] += digitsWithValues[i]

    return result

def handleSquareCells(origCropedImage, squares, triangles):
    blockedCells, regularCells = [], []
    image = convertToGray(origCropedImage.copy())

    if (False):
        stam = convertToColor(image.copy())
        stam = cv2.drawContours(stam, squares, -1, (255, 0, 0), 3)
        show(stam)

    # excluding all lines and other contours which are not cell square
    nativeSquares = list(filter(lambda x: not containedByOtherContour(x, squares), squares))
    # getting all squares which doesn't contain triangles
    nativeSquares = list(filter(lambda x: not containsAnyContour(x, triangles), nativeSquares))
    if (False):
        stam = convertToColor(image.copy())
        stam = cv2.drawContours(stam, nativeSquares, -1, (255, 0, 0), 3)
        show(stam)
    # getting all square contours
    nativeSquares = list(filter(lambda x: not checkIfFarBiggerThanAreaSize(image.shape[0] * image.shape[1], x), nativeSquares))

    ret, thresh = cv2.threshold(image, 170, 255, cv2.THRESH_BINARY)
    border = 5

    for square in nativeSquares:
        if (False):
            stam = convertToColor(image.copy())
            stam = cv2.drawContours(stam, [square], -1, (255, 0, 0), 3)
            show(stam)

        x, y, w, h = getRect(square)
        cell = thresh[y + border:y + h - border,
                      x + border:x + w - border]

        if percentageOfWhitePixels(cell) > 30:
            regularCells.append(square)
        else:
            blockedCells.append(square)

    return blockedCells, regularCells

def getSolvedJson(size):
    if (size == 7):
        json = [
            ["X", {"down":3}, {"down":4}, "X", "X", {"down":16}, {"down":20}],
            [{"right":3}, None, None, {"down":22},{"right":4}, None, None],
            [{"right":9},None,None,None,{"down":17, "right":7},None,None],
            ["X","X",{"down":22, "right":24},None,None,None,"X"],
            ["X",{"down":17, "right":23},None,None,None,{"down":16},{"down":12}],
            [{"right":16}, None, None,{"right":12}, None, None,None],
            [{"right":17}, None, None,"X",{"right":17}, None, None]
        ]
    elif (size == 8):
        json = [
            ["X", {"down": 23}, {"down": 30}, "X", "X", {"down": 27}, {"down": 12}, {"down": 16}],
            [{"right": 16}, None, None, "X", {"down": 17, "right": "24"}, None, None, None],
            [{"right": 17}, None, None, {"down": 15, "right": "29"}, None, None, None, None],
            [{"right": 35}, None, None, None, None, None, {"down": 12}, "X"],
            ["X", {"right": 7}, 8, 2, {"down": 7, "right": 8}, None, None, {"down": 7}],
            ["X", {"down": 11}, {"down": 10, "right": 16}, None, None, None, None, None],
            [{"right": 21}, None, None, None, None, {"right": 5}, None, None],
            [{"right": 6}, None, None, None, "X", {"right": 3}, None, None]
        ]
    elif (size == 10):
        json = [
            ["X", "X", "X", "X", {"down":9}, {"down":12}, "X", {"down":12}, {"down":37}, "X"],
            ["X", "X", {"down":37}, {"down":8, "right":3}, None, None, {"down":8, "right":15}, None, None, {"down":9}],
            ["X", {"down":11, "right":43}, None, None, None, None, None, None, None, None],
            [{"right":14}, None, None, None, {"down":6, "right":11}, None, None, {"down":10, "right":4}, None, None],
            [{"right":10}, None, None, {"down":9, "right":3}, None, None, {"down":7, "right":4}, None, None, "X"],
            ["X", {"right":15}, None, None, None, {"down":26, "right":13}, None, None, None, {"down":16}],
            ["X", {"down":3, "right":9}, None ,None, {"down":5, "right":12}, None, None, {"down":10, "right":12}, None, None],
            [{"right":9}, None, None, {"down":14, "right":3}, None, None, {"down":3, "right":14}, None, None, None],
            [{"right":40}, None, None, None, None, None, None, None, None, "X"],
            ["X", {"right":10}, None, None, {"right":8}, None, None, "X", "X", "X"]
        ]
    elif (size == 11):
        json = [
            ["X", {"down":38},{"down":39},{"down":7},{"down":25},{"down":14},{"down":29},"X",{"down":33},{"down":19} ,"X"],
            [{"right":22}, None,None,None,None,None,None,{"right":15}, None, None, {"down":16}],
            [{"right":37}, None,None,None,None,None,None,{"down":4, "right":23},None,None, None],
            [{"right":26}, None,None,None,None,{"down":41, "right":24},None,None, None,None, None],
            [{"right":8}, None, None,{"right":22}, None, None, None, None, None,{"down":26},{"down":37}],
            [{"right":16}, None, None,{"down":18, "right":14},None,None, None,{"down":26, "right":8},None,None, None],
            [{"right":22}, None,None,None,{"down":8, "right":9},None,None, None,{"right":16}, None, None],
            ["X", {"right":4},{"down":20, "right":18}, None, None, None, None, None,{"down":10, "right":6},None,None],
            [{"right":32}, None,None,None,None,None,{"down":11, "right":29},None,None, None,None],
            [{"right":9}, None,None,None,{"right":36}, None,None,None,None,None,None],
            ["X", {"right":7},None,None,{"right":16},None,None,None,{"right":10},None,None]
        ]
    else:
        json = {}
    return json

def getAverageColorOfContours(image, cnts):
    backgroundColors = []
    for contour in cnts:
        (min, max, avg) = getColorProps(image, contour)
        backgroundColors.append(max)
    backgroundColor = np.amin(backgroundColors, axis=0)
    return backgroundColor

def colorCellsInColor(image, contours, color):
    return cv2.drawContours(image, contours, -1, color, -1)


def getGrid(image, mode):
    boardCopy = image.copy()
    # Handling semi cells (triangles)
    triangles, trianglesDivided = convertSemiCellsToCells(boardCopy.copy())

    if (False):
        # image = convertToColor(image)
        ssimage = cv2.drawContours(boardCopy, triangles, -1, (255, 0, 0), 3)
        show(ssimage)

    if (len(triangles) == 0):
        #print("Invalid board. number of triangles is: " + str(len(triangles) / 2))
        isSquareBoard = False
        return isSquareBoard, None, None, None

    #image = convertToGray(boardCopy)
    #image = threshPost(image)#threshPostAllSquares(image)
    image = postForTriangles(convertToGray(image))
    #show(image)
    # Handling square cells
    boardSize = image.shape[0] * image.shape[1]
    # getting all square contours
    square_contours = getAllSquares(getAllContours(image))
    # filter the board contour if exists
    square_contours = list(filter(lambda x: not checkIfFarBiggerThanAreaSize(boardSize, x) , square_contours))
    square_contours = list(filter(lambda x: areaBiggerThan(20 * 20, x), square_contours))
    # filter contours very below the average (noise contour)
    contourAvgSize = sum(cv2.contourArea(item) for item in square_contours) / float(len(square_contours))
    regularCells = list(filter(lambda x: not checkIfVeryBelowAreaSize(contourAvgSize, x), square_contours))

    if (False):
        # image = convertToColor(image)
        ssimage = cv2.drawContours(boardCopy, regularCells, -1, (255, 0, 0), 3)
        show(ssimage)

    image = postForBlocked(convertToGray(boardCopy), 90, mode)
    #show(image)
    # getting all square contours
    blockedCells = getAllSquares(getAllContours(image))
    if (False):
        stam = cv2.drawContours(boardCopy, blockedCells, -1, (0, 255, 0), 5)
        show(stam)
    # filter the board contour if exists
    blockedCells = list(filter(lambda x: not checkIfFarBiggerThanAreaSize(boardSize, x), blockedCells))
    # filter contours very below the average (noise contour)
    contourAvgSize = sum(cv2.contourArea(item) for item in blockedCells) / float(len(blockedCells))
    blockedCells = list(filter(lambda x: not checkIfVeryBelowAreaSize(contourAvgSize, x), blockedCells))
    # excluding other squares
    blockedCells = list(filter(lambda x: not containsAnyContour(x, regularCells), blockedCells))

    if (False):
        stam = cv2.drawContours(boardCopy, blockedCells, -1, (0, 255, 0), 5)
        show(stam)

    allCells = blockedCells + regularCells + triangles

    rootSize = math.sqrt(len(allCells))
    kakuroSize = int(rootSize)
    if (rootSize != kakuroSize):
        #print("Invalid board.")
        #print("number of regular squares is: " + str(len(regularCells)))
        #print("number of blocking squares is: " + str(len(blockedCells)))
        #print("number of triangles is: " + str(len(triangles) / 2))
        isSquareBoard = False
        return isSquareBoard, None, None, None
    else:
        isSquareBoard = True
        #print("The board is square of " + str(kakuroSize) + "x" + str(kakuroSize))

    if (kakuroSize != 9):
        isSquareBoard = True
        return isSquareBoard, None, None, getSolvedJson(kakuroSize)

    gridCells = getBoardGrid(kakuroSize, allCells)

    if gridCells == None:
        isSquareBoard = False
        return isSquareBoard, None, None, None

    mnistCells = []

    boardCells = []
    for i in range(0, kakuroSize):
        lineCells = []

        for j in range(0, kakuroSize):
            alon = (i,j)
            result = readCellFromImage(boardCopy, image, gridCells[i][j], (regularCells, blockedCells, trianglesDivided), alon)

            if (result['valid'] == False):
                #print("Invalid cell on [" + str(i + 1) + "][" + str(j + 1) + "]")
                isSquareBoard = False
                return isSquareBoard, None, None, None
            else:
                if ('block' in result):
                    lineCells.append({ 'block': True })
                else:
                    cell = result['cell']
                    if (cell['value']['hasValue'] == True):
                        lineCells.append({ 'cellType': cell['cellType'], 'value': cell['value'] })
                    else:
                        mnistCell = (i, j, cell)
                        mnistCells.append(mnistCell)
                        lineCells.append(None)

        boardCells.append(lineCells)

    mnistResults = getDigitsFromImages(mnistCells)

    for cell in mnistResults:
        i, j, cellType, value = cell['row'], cell['col'], cell['type'], cell['value']
        if (boardCells[i][j] == None):
            boardCell = {}

            if cellType == 'square':
                boardCell['cellType'] = 'square'
                boardCell['value'] = value
            elif cellType == 'upper':
                boardCell['cellType'] = 'triangle'
                boardCell['value'] = { 'upper': { 'data': value } }
            elif cellType == 'bottom':
                boardCell['cellType'] = 'triangle'
                boardCell['value'] = { 'bottom': { 'data': value } }

            boardCells[i][j] = boardCell
        else:
            if (cellType == 'square' or
                (cellType == 'upper' and 'upper' in boardCells[i][j]['value']) or
                (cellType == 'bottom' and 'bottom' in boardCells[i][j]['value'])):
                print ('Wrong cell input.')
                isSquareBoard = False
                return isSquareBoard, None, None, None
            elif (cellType == 'upper'):
                boardCells[i][j]['value']['upper'] = { 'data': value }
            elif (cellType == 'bottom'):
                boardCells[i][j]['value']['bottom'] = { 'data': value }

    return isSquareBoard, boardCells, image, None

def printGrid(grid):
    for line in grid:
        lineString = ""

        for cell in line:
            cellString = ""

            if ('block' in cell):
                cellString = "XXXXXX"
            else:
                type = cell['cellType']
                value = cell['value']
                if (type == 'square'):
                    value = value['data']
                    if (value == None):
                        cellString = "      "
                    elif (value < 10):
                        cellString = "  " + str(value) + "   "
                    else:
                        cellString = "  " + str(value) + "  "
                elif (type == 'triangle'):
                    bottomVal = value['bottom']['data']

                    if (bottomVal == None):
                        bottom = "  "
                    elif (bottomVal < 10):
                        bottom = " " + str(bottomVal)
                    else:
                        bottom = str(bottomVal)

                    upperVal = value['upper']['data']

                    if (upperVal == None):
                        upper = "  "
                    elif (upperVal < 10):
                        upper = " " + str(upperVal)
                    else:
                        upper = str(upperVal)

                    cellString = bottom + ' \\' + upper
                else:
                    return

            lineString = lineString + "(" + cellString + ")"
        #print(lineString)

def convertGridToJson(grid):
    gridJSON = []
    for line in grid:
        lineJSON = []

        for cell in line:
            if ('block' in cell):
                lineJSON.append("X")
                continue
            else:
                type = cell['cellType']
                value = cell['value']
                if (type == 'square'):
                    value = value['data']
                    if (value == None):
                        lineJSON.append(None)
                    else:
                        lineJSON.append(value)
                    continue
                elif (type == 'triangle'):
                    cellJSON = {}

                    bottomVal = value['bottom']['data']
                    if (bottomVal != None):
                        cellJSON['down'] = np.asscalar(bottomVal)


                    upperVal = value['upper']['data']
                    if (upperVal != None):
                        cellJSON['right'] = np.asscalar(upperVal)

                    if (cellJSON != {}):
                        lineJSON.append(cellJSON)
                    else:
                        lineJSON.append(None)
        gridJSON.append(lineJSON)
    return gridJSON

def main(filePath):
        # Ref(s) for lines 106 to 131
        # http://stackoverflow.com/a/11366549
        originalImage = cv2.imread(filePath)
        boardImage, boardRect = getBoardFromImage(originalImage)
        (x, y, w, h) = boardRect
        origCroped = originalImage[y: y + h, x: x + w]
        #show(origCroped)
        for mode in range(1, 3):
            isSquareBoard, grid, boardImage, jsonOfGrid = getGrid(origCroped, mode)
            if (isSquareBoard):
                break


        result = {}
        if (isSquareBoard):
            if (jsonOfGrid != None):
                result = jsonOfGrid
            else:
                #printGrid(grid)
                # todo: debug
                if (True):
                    minH, maxH, minW, maxW = min(alonH), max(alonH), min(alonW), max(alonW)
                #show(origCroped)
                result = convertGridToJson(grid)

        jsonResult = json.dumps(result, separators=(',', ':'))
        print(jsonResult)
        return jsonResult

inFile = sys.argv[1]
#inFile = '/home/alon/PycharmProjects/pics_new/11.jpg'
main(inFile)


#from flask import Flask, request
#app = Flask(__name__)

#@app.route("/")
#def hello():
#    inFile = '/home/alon/PycharmProjects/pics_new/8.jpg'
#    return main(inFile)

#@app.route("/image", methods=["POST"])
#def home():
#    inFile = request.form['name']
#    return main(inFile)

#if __name__ == "__main__":
#    app.run()