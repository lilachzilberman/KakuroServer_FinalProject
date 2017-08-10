import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np  # in order to save the confusionMatrix into a file
import math
#import datetime

# other files
from mnist.dataHandler import preProcessImages
from mnist.config import size, isTraining, epochs, batchSize, trainPercentage, learningRate, BASE_PATH

# Constants
tensorboardFolder = BASE_PATH + 'tensorBoard'
confusionsFolder = BASE_PATH + 'confusionMatrix'
modelFilePath = BASE_PATH + 'model/my-model'

#def shuffleData(x, y):
#    zipped = list(zip(x, y)) # https://docs.python.org/2/library/functions.html#zip
#    random.shuffle(zipped) - import random
#    x2, y2 = zip(*zipped) # unzipping
#    return x2, y2

def exportToCsv(filePath, data):
    np.savetxt(filePath, data, delimiter=',')

def saveModel(session):
    modelSaver = tf.train.Saver()
    modelSaver.save(session, modelFilePath)

def getModel(session):
    modelSaver = tf.train.Saver()
    modelSaver.restore(session, modelFilePath)
    return session

# Net Constants
inputLayerSize = size * size
outputLayerSize = 10
sizeAfterLastMaxPool = int(math.ceil(size / 4))  # because of the kernel and striding we chose
dropoutProbability = 0.5

# conv filter is of form: height, width, input_channels, output_channels (Bias size == output_channels)
filterSize = 5
conv1_output_channels = 32
conv2_output_channels = 64


# https://www.tensorflow.org/versions/r0.11/api_docs/python/nn/convolution#conv2d
# https://www.tensorflow.org/api_docs/python/tf/nn/conv2d

# Computes a 2-D convolution given 4-D input and filter tensors.
# Given an input tensor of shape [batch, in_height, in_width, in_channels]
# and a filter   tensor of shape [filter_height, filter_width, in_channels, out_channels], this op performs the following:
# 1. Flattens the filter to a 2-D matrix with shape [filter_height * filter_width * in_channels, output_channels].
# 2. Extracts image patches from the input tensor to form a virtual tensor of shape
#    [batch, out_height, out_width, filter_height * filter_width * in_channels].
# 3. For each patch, right-multiplies the filter matrix and the image patch vector.

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# same convolution and max pooling like MNIST (https://www.tensorflow.org/get_started/mnist/pros)
# Our convolutions uses a stride of one and are zero padded so that the output is the same size as the input.
def conv2d(input, filter):
    # 'SAME' = padding with zeros. 'VALID' = without padding (might lose data)
    # https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network.py
    strides = 1
    return tf.nn.conv2d(input, filter, strides=[1, strides, strides, 1], padding='SAME')

# Our pooling is plain old max pooling over 2x2 blocks.
def max_pool_2x2(value):
    # 'SAME' = padding with zeros. 'VALID' = without padding (might lose data)
    # https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network.py
    k = 2
    return tf.nn.max_pool(value, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

# Creating the net
# Creating the first layer
with tf.name_scope('Input_Layer'):
    x = tf.placeholder(tf.float32, shape=[None, inputLayerSize])
    # we first reshape x to a 4d tensor,
    # with the second and third dimensions corresponding to image width and height,
    # and the final dimension corresponding to the number of color channels.
    layer_1 = tf.reshape(x, [-1, size, size, 1])

# Creating first convolution layer
with tf.name_scope('First_Convolution_Layer'):
    # conv filter [filter_height, filter_width, in_channels, out_channels]
    W_conv1 = weight_variable([filterSize, filterSize, 1, conv1_output_channels])
    # Creating a constant tensor for the bias (Bias size == output_channels)
    b_conv1 = bias_variable([conv1_output_channels])

    with tf.name_scope('Activation_Function'):
        layer_2 = tf.nn.relu(conv2d(layer_1, W_conv1) + b_conv1)

    with tf.name_scope('Max_Pooling'):
        layer_3 = max_pool_2x2(layer_2)

# Creating second convolution layer
with tf.name_scope('Second_Convolution_Layer'):
    # conv filter [filter_height, filter_width, in_channels, out_channels]
    W_conv2 = weight_variable([filterSize, filterSize, conv1_output_channels, conv2_output_channels])
    # Creating a constant tensor for the bias (Bias size == output_channels)
    b_conv2 = bias_variable([conv2_output_channels])

    with tf.name_scope('Activation_Function'):
        layer_4 = tf.nn.relu(conv2d(layer_3, W_conv2) + b_conv2)

    with tf.name_scope('Max_Pooling'):
        layer_5 = max_pool_2x2(layer_4)

# Flattening for fully connected layer
with tf.name_scope('Flattening_for_Fully_Connected_Layer'):
    layer_5_flat = tf.reshape(layer_5, [-1, sizeAfterLastMaxPool * sizeAfterLastMaxPool * conv2_output_channels])

# Creating first fully connected layer
with tf.name_scope('2048_Fully_Connected'):
    fc1_output = 2048
    W_fc1 = weight_variable([sizeAfterLastMaxPool * sizeAfterLastMaxPool * conv2_output_channels, fc1_output])
    # Creating a constant tensor for the bias (Bias size == output_channels)
    b_fc1 = bias_variable([fc1_output])

    with tf.name_scope('Activation_Function'):
        layer_6 = tf.nn.relu(tf.matmul(layer_5_flat, W_fc1) + b_fc1)

# Creating second fully connected layer
with tf.name_scope('1024_Fully_Connected'):
    fc2_output = 1024
    W_fc2 = weight_variable([fc1_output, fc2_output])
    # Creating a constant tensor for the bias (Bias size == output_channels)
    b_fc2 = bias_variable([fc2_output])
    with tf.name_scope('Activation_Function'):
        layer_7 = tf.nn.relu(tf.matmul(layer_6, W_fc2) + b_fc2)

with tf.name_scope('Dropout'):
    keep_prob = tf.placeholder(tf.float32)
    # Dropout is often very effective at reducing overfitting,
    # but it is most useful when training very large neural networks
    layer_8 = tf.nn.dropout(layer_7, keep_prob)

with tf.name_scope('Output_Layer'):
    W_fc3 = weight_variable([fc2_output, outputLayerSize])
    # Creating a constant tensor for the bias (Bias size == output_channels)
    b_fc3 = bias_variable([outputLayerSize])
    y_pred = tf.matmul(layer_8, W_fc3) + b_fc3
    y = tf.placeholder(tf.float32, shape=[None, outputLayerSize])
    # getting entropy for loss function
    #cross_entropy = tf.reduce_sum(
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y))

    # data for TensorBoard
    with tf.name_scope('Session_Summaries'):
        mean = tf.reduce_mean(y)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(y - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.histogram('histogram', y)

tf.summary.scalar('Cross_Entropy', cross_entropy)

with tf.name_scope('Adam_Optimizer'):
    # back propogation definition. minimize function adds operations to minimize the cross_entropy
    optimizer = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(cross_entropy)

with tf.name_scope('Loss_Function'):
    with tf.name_scope('Prediction'):
        prediction = tf.argmax(y_pred, 1)  # the index of the maximum element
        realLabels = tf.argmax(y, 1)  # the index of the maximum element
        correct_prediction = tf.equal(prediction, realLabels)
    with tf.name_scope('Accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

tf.summary.scalar('Accuracy', accuracy)

with tf.name_scope('Confusion_Matrix'):
    confusionMatrix = tf.contrib.metrics.confusion_matrix(prediction, realLabels)
# End of creating the net

def run(imagesToRead=None, isTraining=False):
    if (isTraining):
        with tf.name_scope('Init_Session'):
            sess = tf.Session()
            # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
            #sess = tf.InteractiveSession()
            sess.run(tf.global_variables_initializer())

        # initializing tensorflow summary
        #date = datetime.datetime.now().strftime("%d-%m_%H-%M-%S")
        #print('Started training at:' + date)
        #trainFileName = '{}/lr{}_batch{}_epochs{}_size{}_{}/train'.format(tensorboardFolder, learningRate, batchSize, epochs, size, date)
        #testFileName = '{}/lr{}_batch{}_epochs{}_size{}_{}/test'.format(tensorboardFolder, learningRate, batchSize, epochs, size, date)
        #trainSummary = tf.summary.FileWriter(trainFileName, sess.graph)
        #testSummary = tf.summary.FileWriter(testFileName)
        allSummeries = tf.summary.merge_all()

        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        #mnist = fetch_mldata("MNIST original")
        #images, labels = mnist.train.images[:60000], mnist.train.labels[:60000]
        #number_of_examples = len(images)
        #x = [ex for ex, ey in zip(images, labels)]
        #y = labels
        # x, y = shuffle(x, y, random_state=1)

        #trainLength = int(len(images) * trainPercentage)
        #train_x = images[:trainLength]
        #train_y = labels[:trainLength]
        #test_x = images[trainLength:]
        #test_y = labels[trainLength:]
        #test_x = mnist.test.images
        #test_y = mnist.test.labels
        # Creating test data sets

        #print('Started real training at:' + date)

        for i in range(epochs):
            batch = mnist.train.next_batch(batchSize)
            batch_x = batch[0]
            batch_y = batch[1]

            # choose example
            #ind = random.randint(0, number_of_examples - 1)

            # https://www.tensorflow.org/get_started/summaries_and_tensorboard
            # using optimizer in order to do back propogation
            summary, acc = sess.run([allSummeries, optimizer],
                                    feed_dict={ x: batch_x, y: batch_y, keep_prob: dropoutProbability })

            # Every 100th step, measure accuracy, and write summaries
            if (i%100 == 0):
                #trainSummary.add_summary(summary, i)

                summary, acc, confuse = sess.run([allSummeries, accuracy, confusionMatrix],
                                        feed_dict={ x: batch_x, y: batch_y, keep_prob: 1.0})
                #testSummary.add_summary(summary, i)

                print('The accuracy on epoch {} is: {}'.format(i + 1, acc))

            # Shuffling before next epoch
            #train_x, train_y = shuffleData(train_x, train_y)

        # Running last time to conclude all results (if the number of images doesn't divide by 20
        summary, acc, confuse = sess.run([allSummeries, accuracy, confusionMatrix],
                                         feed_dict={ x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
        print('The total accuracy for test set is: {}'.format(acc))

        # Saving confusion matrix on csv format
        confusionFileName = '{}/lr{}_batch{}_epochs{}_size{}_{}.csv'.format(confusionsFolder, learningRate, batchSize, epochs, size, date)
        exportToCsv(confusionFileName, confuse)

        #testSummary.close()
        #trainSummary.close()

        saveModel(sess)

        #date = datetime.datetime.now().strftime("%d-%m_%H:%M:%S")
        #print('Finished training at:' + date)

    # not training
    else:
        imagesToRead = preProcessImages(imagesToRead, size)

        sess = tf.Session()
        sess = getModel(sess)

        moshe_acc = tf.argmax(y_pred, 1)  # the index of the maximum element
        # Getting predictions for all given images (without dropout since we're not training anymore)
        finalResults = sess.run(moshe_acc, feed_dict={ x: imagesToRead, keep_prob: 1.0})
        #finalResults = [x+1 for x in finalResults]

        return finalResults

        # Saving results
        #mosheFileName = '{}/moshe.csv'.format(confusionsFolder)
        #exportToCsv(mosheFileName, finalResults)