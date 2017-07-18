#==============================================================================
""" 
TensorFlow Assignment 3: Ensemble of Convolutional Neural Nets  
                         Bootstrap Aggregating or Bagging
Instructions:
  1. Create an ensemble of 5 Convoluted Neural Networks
  2. Use TensorFlow Saver() object for saving and retrieving variables
  3. Create random training sets for each Neural Nets
  4. Source Code for plotting images are provided by TA 
  5. Remember to include the bias term in the model!!!
  6. Use AdamOptimizer
  7. Compute accuracy of the model and print out confusion matrix
  
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
import os
import prettytensor as pt 

#==============================================================================
#Preliminary Function and Variable definitions
#==============================================================================
#Loading MNIST Data
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST/', one_hot=True)

# Getting Class Numbers
data.test.cls = np.argmax(data.test.labels, axis=1)
data.validation.cls = np.argmax(data.validation.labels, axis=1)

""" Generation of Random train-test datasets"""

# Combine test-train datasets

combined_images = np.concatenate([data.train.images, data.validation.images], axis=0)
combined_labels = np.concatenate([data.train.labels, data.validation.labels], axis=0)

combined_size = len(combined_images)
train_size = int(0.8 * combined_size)
validation_size = combined_size - train_size

#Function for splitting combined data into test-validation-train sets:
    
def random_training_set():
    # Create a randomized index into the full / combined training-set.
    idx = np.random.permutation(combined_size)

    idx_train = idx[0:train_size]
    idx_validation = idx[train_size:]

    x_train = combined_images[idx_train, :]
    y_train = combined_labels[idx_train, :]

    x_validation = combined_images[idx_validation, :]
    y_validation = combined_labels[idx_validation, :]

    return x_train, y_train, x_validation, y_validation

"""Size definitions"""

# MNIST images are 28 pixels in each dimension.
img_size = 28

# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 1

# Number of classes, one class for each of 10 digits.
num_classes = 10

"""Function for plotting images"""

def plot_images(images,                  # Images to plot, 2-d array.
                cls_true,                # True class-no for images.
                ensemble_cls_pred=None,  # Ensemble predicted class-no.
                best_cls_pred=None):     # Best-net predicted class-no.

    assert len(images) == len(cls_true)

    fig, axes = plt.subplots(3, 3)

    if ensemble_cls_pred is None:
        hspace = 0.3
    else:
        hspace = 1.0
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    for i, ax in enumerate(axes.flat):

        if i < len(images):

            ax.imshow(images[i].reshape(img_shape), cmap='binary')

            if ensemble_cls_pred is None:
                xlabel = "True: {0}".format(cls_true[i])
            else:
                msg = "True: {0}\nEnsemble: {1}\nBest Net: {2}"
                xlabel = msg.format(cls_true[i],
                                    ensemble_cls_pred[i],
                                    best_cls_pred[i])

            ax.set_xlabel(xlabel)

        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()
    
#==============================================================================
# TensorFlow Computational Graph
#==============================================================================
"""Placeholder Variables"""

x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)


"""Construction of Convolutional Neural Network"""

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
        
        
"""Optimizer"""

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

""" Performance Measurement"""

y_pred_cls = tf.argmax(y_pred, dimension=1)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


"""Saving Variables"""

saver = tf.train.Saver(max_to_keep=100)
save_dir = 'checkpoints/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
def get_save_path(net_number):
    return save_dir + 'network' + str(net_number)

#==============================================================================
# TensorFlow Sesssion Run
#==============================================================================

session = tf.Session()

def init_variables():                           #Initializing Variables
    session.run(tf.initialize_all_variables())

"""Function to create random training batch"""
   
train_batch_size = 64

def random_batch(x_train, y_train):
    # Total number of images in the training-set.
    num_images = len(x_train)

    # Create a random index into the training-set.
    idx = np.random.choice(num_images,
                           size=train_batch_size,
                           replace=False)

    # Use the random index to select random images and labels.
    x_batch = x_train[idx, :]  # Images.
    y_batch = y_train[idx, :]  # Labels.

    # Return the batch.
    return x_batch, y_batch

""" Function to optimize iterations"""

def optimize(num_iterations, x_train, y_train):
   
    start_time = time.time()

    for i in range(num_iterations):

        x_batch, y_true_batch = random_batch(x_train, y_train)
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}
        session.run(optimizer, feed_dict=feed_dict_train)

        if i % 100 == 0:
            acc = session.run(accuracy, feed_dict=feed_dict_train)
            msg = "Optimization Iteration: {0:>6}, Training Batch Accuracy: {1:>6.1%}"
            print(msg.format(i + 1, acc))

    end_time = time.time()
    time_dif = end_time - start_time
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
    
"""Creating an Ensemble of 5 Neural Networks"""

num_networks = 5

num_iterations = 10000

if True:
    for i in range(num_networks):
        print("Neural network: {0}".format(i))
        x_train, y_train, _, _ = random_training_set()

        session.run(tf.global_variables_initializer())

        optimize(num_iterations=num_iterations,
                 x_train=x_train,
                 y_train=y_train)

        saver.save(sess=session, save_path=get_save_path(i))

        print()

"""Functions for calculating and predicting classifications"""

batch_size = 256

def predict_labels(images):

    num_images = len(images)

    pred_labels = np.zeros(shape=(num_images, num_classes),
                           dtype=np.float)
    i = 0

    while i < num_images:

        j = min(i + batch_size, num_images)

        feed_dict = {x: images[i:j, :]}

        pred_labels[i:j] = session.run(y_pred, feed_dict=feed_dict)

        i = j

    return pred_labels

#Function for a Boolean array of whether the predicted class is correct
def correct_prediction(images, labels, cls_true):
    pred_labels = predict_labels(images=images)
    cls_pred = np.argmax(pred_labels, axis=1)
    correct = (cls_true == cls_pred)
    return correct

#Function for correctness of the test-set classification
def test_correct():
    return correct_prediction(images = data.test.images,
                              labels = data.test.labels,
                              cls_true = data.test.cls)
    
#Function for correctness of the cross-validation-set classification
def validation_correct():
    return correct_prediction(images = data.validation.images,
                              labels = data.validation.labels,
                              cls_true = data.validation.cls)

"""Functions for classification Accuracy"""

def classification_accuracy(correct):
    return correct.mean()

# Function for test-set accuracy
def test_accuracy():
    correct = test_correct()
    return classification_accuracy(correct)

#Function for cross-validation set accuracy
def validation_accuracy():
    correct = validation_correct()
    return classification_accuracy(correct)

"""Results and Analysis"""

#Function for calculating predicted labels
def ensemble_predictions():
    # Empty list of predicted labels for each of the neural networks.
    pred_labels = []

    # Classification accuracy on the test-set for each network.
    test_accuracies = []

    # Classification accuracy on the validation-set for each network.
    val_accuracies = []

    for i in range(num_networks):
        # Reload the variables into the TensorFlow graph.
        saver.restore(sess=session, save_path=get_save_path(i))

        # Calculate the classification accuracy on the test-set.
        test_acc = test_accuracy()

        # Append the classification accuracy to the list.
        test_accuracies.append(test_acc)

        # Calculate the classification accuracy on the validation-set.
        val_acc = validation_accuracy()

        # Append the classification accuracy to the list.
        val_accuracies.append(val_acc)

        msg = "Network: {0}, Accuracy on Validation-Set: {1:.4f}, Test-Set: {2:.4f}"
        print(msg.format(i, val_acc, test_acc))

        # Calculate the predicted labels for the images in the test-set.
        pred = predict_labels(images=data.test.images)

        # Append the predicted labels to the list.
        pred_labels.append(pred)
    
    return np.array(pred_labels), \
           np.array(test_accuracies), \
           np.array(val_accuracies)
           
pred_labels, test_accuracies, val_accuracies = ensemble_predictions()

#Summarize Classification accuracies on the test-set for the ensemble
print("Mean test-set accuracy: {0:.4f}".format(np.mean(test_accuracies)))
print("Min test-set accuracy:  {0:.4f}".format(np.min(test_accuracies)))
print("Max test-set accuracy:  {0:.4f}".format(np.max(test_accuracies)))
pred_labels.shape

"""Ensemble Predictions"""

#Method used = take average of predicted labels of all the ensembles

ensemble_pred_labels = np.mean(pred_labels, axis=0)  #(10000,10)
ensemble_cls_pred = np.argmax(ensemble_pred_labels, axis=1) #(10000,)

#Boolean array of correctnesss of test-set classification by ensemble
ensemble_correct = (ensemble_cls_pred == data.test.cls)
ensemble_incorrect = np.logical_not(ensemble_correct)

print("No. of images in test-set correctly classified by ensemble: {} \n"
      .format(np.sum(ensemble_correct)))

session.close()