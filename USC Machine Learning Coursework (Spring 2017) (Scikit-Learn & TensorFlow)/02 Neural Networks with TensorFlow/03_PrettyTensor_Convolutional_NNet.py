#==============================================================================
""" 
TensorFlow Assignment 3: Repeat Convolutional Neural Net assingment 
                         with PrettyTensor
Instructions:
  1. Modify only TensorFlow graph part of the previous code
  2. Use AdamOptimizer
  3. Compute accuracy of the model and print out confusion matrix
  
Scikit-Learn Usage Reference: 
    Scikit-Learn: Machine Learning in Python,
    Pedregosa et al., JMLR 12, pp. 2825-2830, 2011

@author: Dhineshkumar"""
#==============================================================================


import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import prettytensor as pt

#==============================================================================
#Preliminary Function and Variable definitions
#=============================================================================

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST/', one_hot=True)

data.test.cls = np.argmax(data.test.labels, axis=1)

"""Size definitions"""

img_size = 28

img_size_flat = img_size * img_size

img_shape = (img_size, img_size)

num_channels = 1 # Number of colour channels for the images: 
                    #1 channel for gray-scale.

num_classes = 10 # Number of classes, one class for each of 10 digits.

"""Function for plotting images"""

def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

"""Placeholder Variables"""
    
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')

x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])

y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')

y_true_cls = tf.argmax(y_true, dimension=1)

def new_weights(shape):
    
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    
    return tf.Variable(tf.constant(0.05, shape=[length]))

"""Model Definition"""

x_pretty = pt.wrap(x_image)
 
with pt.defaults_scope(activation_fn=tf.nn.relu):
    y_pred, loss = x_pretty.\
        conv2d(kernel=5, depth=16, name='layer_conv1').\
        max_pool(kernel=2, stride=2).\
        conv2d(kernel=5, depth=36, name='layer_conv2').\
        max_pool(kernel=2, stride=2).\
        flatten().\
        fully_connected(size=128, name='layer_fc1').\
        softmax_classifier(num_classes=num_classes, labels=y_true)
        
""" Retrieving an existing variable named 'weights'"""
def get_weights_variable(layer_name):

    with tf.variable_scope(layer_name, reuse=True):
        variable = tf.get_variable('weights')

    return variable


weights_conv1 = get_weights_variable(layer_name='layer_conv1')
weights_conv2 = get_weights_variable(layer_name='layer_conv2')

"""Optimization Method"""

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

"""Performance Metrics"""

y_pred_cls = tf.argmax(y_pred, dimension=1)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#==============================================================================
# TensorFlow Session
#==============================================================================

session = tf.Session()

session.run(tf.global_variables_initializer())

train_batch_size = 64

# Counter for total number of iterations performed so far.

total_iterations = 0

def optimize(num_iterations):
   
    global total_iterations

    start_time = time.time()

    for i in range(total_iterations,
                   total_iterations + num_iterations):

        x_batch, y_true_batch = data.train.next_batch(train_batch_size)

        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        session.run(optimizer, feed_dict=feed_dict_train)

        if i % 100 == 0:

            acc = session.run(accuracy, feed_dict=feed_dict_train)

            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

            print(msg.format(i + 1, acc))

    total_iterations += num_iterations

    end_time = time.time()

    time_dif = end_time - start_time

    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
    
def plot_example_errors(cls_pred, correct):

    incorrect = (correct == False)

    images = data.test.images[incorrect]

    cls_pred = cls_pred[incorrect]

    cls_true = data.test.cls[incorrect]
    
    # Plotting the first 9 images.
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])
    
def plot_confusion_matrix(cls_pred):

    cls_true = data.test.cls

    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    print(cm)

    plt.matshow(cm)

    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    plt.show()
    
# Split the test-set into smaller batches of this size.
test_batch_size = 256

def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):

    num_test = len(data.test.images)

    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    i = 0

    while i < num_test:

        j = min(i + test_batch_size, num_test)

        images = data.test.images[i:j, :]

        labels = data.test.labels[i:j, :]

        feed_dict = {x: images,
                     y_true: labels}

        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        i = j

    cls_true = data.test.cls

    correct = (cls_true == cls_pred)

    correct_sum = correct.sum()

    acc = float(correct_sum) / num_test

    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)
        
print_test_accuracy()

optimize(num_iterations=10000)

print_test_accuracy(show_example_errors=True,
                    show_confusion_matrix=True)

session.close()

