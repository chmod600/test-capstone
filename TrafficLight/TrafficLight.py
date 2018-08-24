from os import listdir
import matplotlib.image as mpimg
import numpy as np
from skimage.transform import resize
import pickle

X_train2 = []
y_train2 = []

path = "traffic_lights/"
image=mpimg.imread(path + "/unknown.png")
image = resize(image, (32, 32))
for x in range(0, 3200):
    X_train2.append(image)
    y_train2.append(0) 
  
path = "traffic_lights/red"
for f in listdir(path):
  image=mpimg.imread(path + "/" + f)
  image = resize(image, (32, 32))
  for x in range(0, 10):
    X_train2.append(image)
    y_train2.append(1)
 
path = "traffic_lights/green"
for f in listdir(path):
  image=mpimg.imread(path + "/" + f)
  image = resize(image, (32, 32))
  for x in range(0,10):
    X_train2.append(image)
    y_train2.append(2)

X_train = np.array(X_train2)
y_train = np.array(y_train2)

#info
import numpy as np

n_train = len(X_train)
image_shape = X_train[0].shape
n_classes = len(np.unique(y_train))

print("Number of training examples =", n_train)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

#from skimage import exposure
#X_train = exposure.equalize_hist(X_train)

#normalize
X_train = (X_train-0.5)/0.5

from tensorflow.contrib.layers import flatten
import tensorflow as tf

def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    filter = tf.Variable(tf.truncated_normal([5, 5, 3, 6], mean = mu, stddev = sigma))
    bias = tf.Variable(tf.zeros(6))
    conv_layer = tf.add(tf.nn.conv2d(x, filter, strides=[1, 1, 1, 1], padding='VALID'), bias, name="conv1")
    
    # TODO: Activation.
    conv_layer = tf.nn.relu(conv_layer, name="conv2")

    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    ksize = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
    padding = 'VALID'
    conv_layer = tf.nn.max_pool(conv_layer, ksize, strides, padding, name="conv3")

    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    filter = tf.Variable(tf.truncated_normal([5, 5, 6, 16], mean = mu, stddev = sigma))
    bias = tf.Variable(tf.zeros(16))
    conv_layer = tf.add(tf.nn.conv2d(conv_layer, filter, strides=[1, 1, 1, 1], padding='VALID'), bias, name="conv4")

    # TODO: Activation.
    conv_layer = tf.nn.relu(conv_layer, name="conv5")

    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    ksize = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
    padding = 'VALID'
    conv_layer = tf.nn.max_pool(conv_layer, ksize, strides, padding, name="conv6")

    # TODO: Flatten. Input = 5x5x16. Output = 400.
    f0 = flatten(conv_layer)
    
    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc_w = tf.Variable(tf.truncated_normal(shape=(400, 200), mean = mu, stddev = sigma))
    fc_b = tf.Variable(tf.zeros(200))
    fc = tf.matmul(f0, fc_w) + fc_b
    
    # TODO: Activation.
    fc = tf.nn.sigmoid(fc)
    #fc = tf.nn.dropout(fc, 0.5)

    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc_w = tf.Variable(tf.truncated_normal(shape=(200, 80), mean = mu, stddev = sigma))
    fc_b = tf.Variable(tf.zeros(80))
    fc = tf.matmul(fc, fc_w) + fc_b

    # TODO: Activation.
    fc = tf.nn.sigmoid(fc)
    #fc = tf.nn.dropout(fc, 0.6)
    
    # TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
    fc_w = tf.Variable(tf.truncated_normal(shape=(80, 3), mean = mu, stddev = sigma))
    fc_b = tf.Variable(tf.zeros(3))
    logits = tf.matmul(fc, fc_w) + fc_b   
    
    return logits

x = tf.placeholder(tf.float32, (None, 32, 32, 3), name="x")
y = tf.placeholder(tf.int32, (None), name="y")
one_hot_y = tf.one_hot(y, n_classes)

# Train
rate = 0.005

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()
BATCH_SIZE = 128
def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

import tensorflow as tf
from sklearn.utils import shuffle

EPOCHS = 10

print(rate)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...\n")
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        training_accuracy = evaluate(X_train, y_train)
        print("EPOCH {} ...".format(i+1))
        print("Training Accuracy = {:.3f}".format(training_accuracy))
        print()
    
    saver.save(sess, './lenet')
    tf.saved_model.simple_save(sess, "outModel", inputs={"input": x}, outputs={"output": logits})
    print("Model saved")

