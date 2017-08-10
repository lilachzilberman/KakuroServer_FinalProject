#from PIL import Image
import numpy as np
#import pickle

from mnist.config import BASE_PATH

from imageHelper import show

ROOT_PATH = BASE_PATH + 'data/'
MOSHE_PATH = BASE_PATH + 'moshe_data/'
IMAGES_FILE = BASE_PATH + 'imagesReshaped{}'
LABELS_FILE = BASE_PATH + 'labels'

EMPTY_CELL = 0.
VALUE_CELL = 1.

def convertImageToArray(image):
    #arr = np.array(image.getdata(), np.float32)
    return np.multiply(image, 1 / 255).tolist()

def convertNDArrayToArray(image):
    arr = np.array(image.flatten(), np.float32)
    return np.multiply(arr, 1 / 255).tolist()


# get both train and test images from file system
#from os import listdir
#import os.path
#def getData(size):
#    x = [] # data
#    y = [] # labels

#    imagesFileName = IMAGES_FILE.format(size)
#    labelsFileName = LABELS_FILE

    # If we already computed the files
#    if (os.path.exists(imagesFileName) and os.path.exists(labelsFileName)):
#        with open(imagesFileName, 'rb') as file:
#            x = pickle.load(file)
#        with open(labelsFileName, 'rb') as file:
#            y = pickle.load(file)
#    else:
#        folders = listdir(ROOT_PATH)
#        for folderName in folders:
#            images = listdir(ROOT_PATH + folderName)
#            for imageName in images:
#                arr = [EMPTY_CELL] * 51               # initializing all cells to be empty
#                arr[int(folderName) - 1] = VALUE_CELL # putting value in the right cell
#                y.append(arr)                         # adding the label to the other labels array

#                imageFullPath = ROOT_PATH + folderName + '/' + imageName
#                image = Image.open(imageFullPath).convert('L') # convert image to B&W (http://stackoverflow.com/questions/18777873/convert-rgb-to-black-or-white)
#                imageSize = size, size
#                image = image.resize(imageSize)
#                imageAsArray = convertImageToArray(image)
#                x.append(imageAsArray) # adding the data

#        with open(imagesFileName, 'wb') as file:
#            pickle.dump(x, file)
#        with open(labelsFileName, 'wb') as file:
#            pickle.dump(y, file)

#    return x, y

def preProcessImages(images, size):
    #TODO: make is faster!
    result = []
    for image in images:
        #image1 = Image.fromarray(image).convert('L')
        #imageSize = size, size
        #image1 = image1.resize(imageSize)
        imageAsArray = convertNDArrayToArray(image)
        result.append(imageAsArray)
    return result

# get final images from file system
def getMosheData(size):

    x = [] # data

    images = listdir(MOSHE_PATH)
    for imageName in images:

        imageFullPath = MOSHE_PATH + imageName
        image = Image.open(imageFullPath).convert('L') # convert image to B&W (http://stackoverflow.com/questions/18777873/convert-rgb-to-black-or-white)
        imageSize = size, size
        image = image.resize(imageSize)
        imageAsArray = convertImageToArray(image)
        x.append(imageAsArray) # adding the data

    return x, images