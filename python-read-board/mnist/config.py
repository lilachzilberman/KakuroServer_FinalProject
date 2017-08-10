import os
BASE_PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
isTraining = False
size = 28 # Size for the image to be trained
epochs = 20000  # Set number of epochs
batchSize = 500  # batch size
learningRate = 1e-4 # learning rate
trainPercentage = 0.85