"""
Project 4 - CNNs
CS1430 - Computer Vision
Brown University
"""

# size of fer images
img_size = 48

# The number of emotions (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).
category_num = 7

# Sample size for calculating the mean and standard deviation of the
# training data. This many images will be randomly seleted to be read
# into memory temporarily.
preprocess_sample_size = 400

# Training parameters

# num_epochs is the number of epochs. If you experiment with more
# complex networks you might need to increase this. Likewise if you add
# regularization that slows training.
num_epochs = 50

# batch_size defines the number of training examples per batch:
# You don't need to modify this.
batch_size = 10

# learning_rate is a critical parameter that can dramatically affect
# whether training succeeds or fails. For most of the experiments in this
# project the default learning rate is safe.
learning_rate = 1e-4

# Momentum on the gradient (if you use a momentum-based optimizer)
momentum = 0.01
