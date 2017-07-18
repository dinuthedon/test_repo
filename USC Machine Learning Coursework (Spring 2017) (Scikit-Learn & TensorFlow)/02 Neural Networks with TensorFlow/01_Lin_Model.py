#==============================================================================
""" 
TensorFlow Assignment 1: Simple Linear Model with MNIST dataset 
                         for hand-written numbers image recognition
Instructions:
  1. Labels of training and test data sets are One-Hot encoded
  2. Source Code for plotting images are provided by TA
  3. Remember to include the bias term in the model!!!
  4. Use GradientDescentOptimizer
  5. Compute accuracy of the model and print out confusion matrix
  
Scikit-Learn Usage Reference: 
    Scikit-Learn: Machine Learning in Python,
    Pedregosa et al., JMLR 12, pp. 2825-2830, 2011

@author: Dhineshkumar"""
#==============================================================================

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix

#==============================================================================
#Preliminary Function and Variable definitions
#=============================================================================

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("data/MNIST/", one_hot=True)

# Getting Class Numbers
data.test.cls = np.array([label.argmax() for label in data.test.labels])

"""Size definitions"""

img_size = 28

img_size_flat = img_size * img_size

img_shape = (img_size, img_size)

"""Function for plotting images"""

def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
      
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        ax.set_xlabel(xlabel)
        
        ax.set_xticks([])
        ax.set_yticks([])

num_classes = 10 # Number of classes, one class for each of 10 digits.

#==============================================================================
# TensorFlow Computational Graph
#==============================================================================

"""Placeholder Variables"""

x = tf.placeholder(tf.float32, [None, img_size_flat])

y_true = tf.placeholder(tf.float32, [None, num_classes])

y_true_cls = tf.placeholder(tf.int64, [None])

"""Model parameters"""

weights = tf.Variable(tf.zeros([img_size_flat, num_classes]))

biases = tf.Variable(tf.zeros([num_classes]))

"""Model Definition"""

logits = tf.matmul(x, weights) + biases

y_pred = tf.nn.softmax(logits)

y_pred_cls = tf.argmax(y_pred, dimension=1)

""" Cost function to be optimized"""

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, 
                                                        labels=y_true)

cost = tf.reduce_mean(cross_entropy)

"""Optimization Method"""

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost)

"""Performance Metrics"""

correct_prediction = tf.equal(y_pred_cls, y_true_cls)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#==============================================================================
# TensorFlow Session
#==============================================================================

session = tf.Session() # Creeate a TF session

session.run(tf.global_variables_initializer()) # Initialize Variables

batch_size = 1000

def optimize(num_iterations):
    for i in range(num_iterations):

        x_batch, y_true_batch = data.train.next_batch(batch_size)

        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        session.run(optimizer, feed_dict=feed_dict_train)

"""Function to show performance"""

feed_dict_test = {x: data.test.images,
                  y_true: data.test.labels,
                  y_true_cls: data.test.cls}

"""Print out Accuracy"""

def print_accuracy():

    acc = session.run(accuracy, feed_dict=feed_dict_test)
    
    print("Accuracy on test-set: {0:.1%}".format(acc))
    
"""Print and plot Confusion Matrix using SciKit """

def print_confusion_matrix():
    
    # Get the true classifications for the test-set.
    cls_true = data.test.cls
    
    # Get the predicted classifications for the test-set.
    cls_pred = session.run(y_pred_cls, feed_dict=feed_dict_test)

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    plt.tight_layout()
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
"""Function for printing misclassified examples"""

def plot_example_errors():
    # Use TensorFlow to get a list of boolean values
    # whether each test-image has been correctly classified,
    # and a list for the predicted class of each image.
    correct, cls_pred = session.run([correct_prediction, y_pred_cls],
                                    feed_dict=feed_dict_test)

    # Negate the boolean array.
    incorrect = (correct == False)
    
    # Get the images from the test-set that have been
    # incorrectly classified.
    images = data.test.images[incorrect]
    
    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = data.test.cls[incorrect]
    
    # Plot the first 9 images.
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])
    
"""Function to plot the model weights"""

def plot_weights():
    # Get the values for the weights from the TensorFlow variable.
    w = session.run(weights)
    
    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(w)
    w_max = np.max(w)

    # Create figure with 3x4 sub-plots,
    # where the last 2 sub-plots are unused.
    fig, axes = plt.subplots(3, 4)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Only use the weights for the first 10 sub-plots.
        if i<10:
            # Get the weights for the i'th digit and reshape it.
            # Note that w.shape == (img_size_flat, 10)
            image = w[:, i].reshape(img_shape)

            # Set the label for the sub-plot.
            ax.set_xlabel("Weights: {0}".format(i))

            # Plot the image.
            ax.imshow(image, vmin=w_min, vmax=w_max, cmap='seismic')

        # Remove ticks from each sub-plot.
        ax.set_xticks([])
        ax.set_yticks([])

"""TensorFlow Execution:"""
      
print_accuracy() # Accuracy Before learning

plot_example_errors()

# Iteration 1

optimize(num_iterations=1)

print_accuracy()

plot_example_errors()

plot_weights()


# We have already performed 1 iteration.

optimize(num_iterations=9) # In order to perform total 10 operations

print_accuracy()

plot_example_errors()

plot_weights()


# We have already performed 10 iterations.

optimize(num_iterations=990)

print_accuracy()

plot_example_errors()

plot_weights()

print_confusion_matrix()

session.close() 