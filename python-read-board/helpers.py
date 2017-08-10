import cv2
import numpy as np
import math

def getAllContours(image):
    _, contours, h = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# inner function
def make_it_square(image, side_length=306):
    return cv2.resize(image, (side_length, side_length))

def cut_out_sudoku_puzzle(image, contour):
    x, y, w, h = getRect(contour)
    image = image[y:y + h, x:x + w]
    return make_it_square(image, min(image.shape))

def thresholdify(img, kernel=31, value=7):
    img = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, kernel,value)#3 #TODO: was 11,10 or 11,7 or 5,2
    return 255 - img

def dilate(image):
    kernel = np.empty((3, 3), 'uint8')
    kernel[0][0] = 0
    kernel[0][1] = 1
    kernel[0][2] = 0
    kernel[1][0] = 1
    kernel[1][1] = 1
    kernel[1][2] = 1
    kernel[2][0] = 0
    kernel[2][1] = 1
    kernel[2][2] = 0
    dilated = cv2.dilate(image, kernel)
    return dilated

def new_dilate(image, size):
    structure_elem = cv2.getStructuringElement(cv2.MORPH_OPEN, (size, size))
    dilated = cv2.dilate(image, structure_elem, iterations=1)
    return dilated

def approx(cnt):
    peri = cv2.arcLength(cnt, True)
    app = cv2.approxPolyDP(cnt, 0.01 * peri, True)
    return app
# are the same?
def getContourApprox(contour):
    peri = cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, 0.02 * peri, True)

def getRect(contour):
    return cv2.boundingRect(contour)

def getAllSquares(contours):
    return list(filter(lambda x: isSquare(x), contours))

def getAllTriangles(contours):
    return list(filter(lambda x: isTriangle(x), contours))

#http://www.pyimagesearch.com/2016/02/08/opencv-shape-detection/
def isSquare(contour):
    approx = getContourApprox(contour)
    if (len(approx) == 4):
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)

        # a square will have an aspect ratio that is approximately
        # equal to one, otherwise, the shape is a rectangle
        # return (ar >= 0.85 and ar <= 1.15)
        return True
    else:
        return False

#http://www.pyimagesearch.com/2016/02/08/opencv-shape-detection/
def isTriangle(contour):
    approx = getContourApprox(contour)
    return (len(approx) == 3)

def calcDistance(pointSrc, pointDst):
    X2 = math.pow(pointSrc[0] - pointDst[0], 2)
    Y2 = math.pow(pointSrc[1] - pointDst[1], 2)
    return math.sqrt(X2 + Y2)

# vertex of triangle

def getMiddleVertex(approx, center):
    minDistance = min(calcDistance(v[0], (center[0], center[1])) for v in approx)
    middleVertex = list(filter(lambda x: calcDistance(x[0], (center[0], center[1])) == minDistance, approx))[0]
    isUpper = (middleVertex[0][0] > center[0])
    return middleVertex, isUpper

def getLeftVertex(approx, middle):
    approxWithoutMiddle = list(filter(lambda x: (x[0][0] != middle[0][0] or x[0][1] != middle[0][1]) , approx))
    leftX = min(v[0][0] for v in approxWithoutMiddle)
    leftVertex = list(filter(lambda x: x[0][0] == leftX, approxWithoutMiddle))
    return leftVertex[0]

def getRightVertex(approx, middle):
    approxWithoutMiddle = list(filter(lambda x: (x[0][0] != middle[0][0] or x[0][1] != middle[0][1]), approx))
    leftY = max(v[0][1] for v in approxWithoutMiddle)
    leftVertex = list(filter(lambda x: x[0][1] == leftY, approxWithoutMiddle))
    return leftVertex[0]

# end vertex of triangle functions

def getContourCenter(contour):
    M = cv2.moments(contour)
    cX = int((M["m10"] / M["m00"]))
    cY = int((M["m01"] / M["m00"]))
    return (cX, cY)

def isCenterEqual(center, contour):
    newCenter = getContourCenter(contour)
    return (newCenter[0] == center[0] and newCenter[1] == center[1])

def getTopLeft(upperLeftVertex, lowerLeftVertex):
    x = min(v[0] for v in [upperLeftVertex[0], lowerLeftVertex[0]])
    y = min(v[1] for v in [upperLeftVertex[0], lowerLeftVertex[0]])
    return x, y

def getBottomRight(upperRightVertex, lowerRightVertex):
    x = max(v[0] for v in [upperRightVertex[0], lowerRightVertex[0]])
    y = max(v[1] for v in [upperRightVertex[0], lowerRightVertex[0]])
    return x, y

def findContourAndRectOfPoint(point, cntsAndRects):
    for (cnt, rect) in cntsAndRects:
        # the point is inside the contour
        if (isPointInContour(point, cnt)):
            return (cnt, rect)
    return None

def containsAnyContour(source, list):
    for dst in list:
        dstCenter = getContourCenter(dst)
        if (isPointInContour(dstCenter, source)):
            return True
    return False

def containedByOtherContour(source, list):
    sourceCenter = getContourCenter(source)
    sourceArea = cv2.contourArea(source)
    for dst in list:
        dstArea = cv2.contourArea(dst)
        if (isPointInContour(sourceCenter, dst) and sourceArea < dstArea):
            return True
    return False

def isPointInContour(point, contour):
    r = cv2.pointPolygonTest(contour, point, False)
    # the point is inside the contour
    return (r == 1)

def invertProcess(image):
    image = cv2.adaptiveThreshold(image.astype(np.uint8), 150, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 7)  # 3 #TODO: was 11,10 or 11,7 or 5,2
    image = cv2.GaussianBlur(image, (5, 5), 0)
    #show(image)
    return image

def threshForSquares(image):
    ret, thresh = cv2.threshold(image, 70, 255, cv2.THRESH_BINARY)
    image = 255 - thresh
    return image

def postForBlocked(image, color, mode):
    ret, thresh = cv2.threshold(image, color, 255, cv2.THRESH_BINARY)
    image = 255 - thresh
    image = cv2.bilateralFilter(image, 9, 75, 75)
    if (mode == 1):
        image = cv2.GaussianBlur(image, (5, 5), 0)
    return image

def postForTriangles(image):
    ret, thresh = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY)
    image = 255 - thresh
    #thresh = cv2.adaptiveThreshold(image.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                cv2.THRESH_BINARY, 31, 2)  # 3 #TODO: was 11,10 or 11,7 or 5,2
#    image = 255 - thresh
    image = cv2.GaussianBlur(image, (5, 5), 0)
    #show(image)
    return image

    ret, thresh = cv2.threshold(image, 160, 255, cv2.THRESH_BINARY)
    image = 255 - thresh
    #image = cv2.GaussianBlur(image, (5, 5), 0)
    # eroding
    kernel = np.ones((7, 7), np.uint8)
    image = cv2.erode(image, kernel, iterations=2)
    #show(image)
    image = cv2.GaussianBlur(image, (7, 7), 0)
    #show(image)
    return image

def threshPostAllSquares(image):
    ret, thresh = cv2.threshold(image, 160, 255, cv2.THRESH_BINARY)
    image = 255 - thresh
    image = cv2.GaussianBlur(image, (9, 9), 0)
    # eroding
    kernel = np.ones((3, 3), np.uint8)
    image = cv2.erode(image, kernel, iterations=2)
    return image

def threshForBlock(image):
    image = cv2.adaptiveThreshold(image.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 5, 1)  # 3 #TODO: was 11,10 or 11,7 or 5,2
    return image
    #ret, thresh = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY_INV)
    #return 255 - thresh

def threshPost(image):
    image = cv2.adaptiveThreshold(image.astype(np.uint8), 160, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 11, 7)  # 3 #TODO: was 11,10 or 11,7 or 5,2
    #show(image)
    image = cv2.GaussianBlur(image, (9, 9), 0)
    #show(image)
    return image
    #image = cv2.bilateralFilter(image, 11, 17, 17)
    #image = cv2.Canny(image, 30, 200)
    #ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# irrelevant
def thresholdifyNew(img):
    img = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, 5,2)#3 #TODO: was 11,10 or 11,7 or 5,2
    return 255 - img

def preProcessNew(image):
    image = cv2.GaussianBlur(image, (11, 11), 0)
    thresh = thresholdifyNew(image)
    dilated = dilate(thresh)
    return (255 - dilated)

def get_rectangle_corners(cnt):
    pts = cnt.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point has the smallest sum whereas the
    # bottom-right has the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # compute the difference between the points -- the top-right
    # will have the minumum difference and the bottom-left will
    # have the maximum difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def warp_perspective(rect, grid):
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

    # ...and now for the height of our new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

    # take the maximum of the width and height values to reach
    # our final dimensions
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))

    # construct our destination points which will be used to
    # map the screen to a top-down, "birds eye" view
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # calculate the perspective transform matrix and warp
    # the perspective to grab the screen
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(grid, M, (maxWidth, maxHeight))
    return make_it_square(warp)

def straighten(image):
    #print('Straightening image...')
    largest = largest4SideContour(image.copy())
    if (largest is None):
        return image
    app = approx(largest)
    corners = get_rectangle_corners(app)
    sudoku = warp_perspective(corners, image)
    return sudoku

def largest4SideContour(image):
    _, contours, h = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours[:min(5, len(contours))]:
        if len(approx(cnt)) == 4:
            return cnt
    return None