"""
Project 4 - CNNs
CS1430 - Computer Vision
Brown University
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from model import Model
import hyperparameters as hp
from pp import Datasets
from tensorboard_utils import ImageLabelingLogger, ConfusionMatrixLogger

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def parse_args():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="Let's train some neural nets!")
    parser.add_argument(
        '--data',
        default='fer',
        help='Image datasets to build model on')
    parser.add_argument(
        '--load-checkpoint',
        default=None,
        help='''Path to model checkpoint file (should end with the
        extension .h5). Checkpoints are automatically saved when you
        train your model. If you want to continue training from where
        you left off, this is how you would load your weights. ''')
    parser.add_argument(
        '--confusion',
        action='store_true',
        help='''Log a confusion matrix at the end of each
        epoch (viewable in Tensorboard). This is turned off
        by default as it takes a little bit of time to complete.''')
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='''Skips training and evaluates on the test set once.
        You can use this to test an already trained model by loading
        its checkpoint.''')

    return parser.parse_args()

def train(model, datasets, checkpoint_path):
    """ Training routine. """

    # Keras callbacks for training
    callback_list = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path + \
                    "weights.e{epoch:02d}-" + \
                    "acc{val_sparse_categorical_accuracy:.4f}.h5",
            monitor='val_sparse_categorical_accuracy',
            save_best_only=True,
            save_weights_only=True),
        tf.keras.callbacks.TensorBoard(
            update_freq='batch',
            profile_batch=0),
        # ImageLabelingLogger(datasets)
    ]

    # Include confusion logger in callbacks if flag set
    # if ARGS.confusion:
    #     callback_list.append(ConfusionMatrixLogger(datasets))

    # Begin training
    print(train_x.shape)
    model.fit(
        x=datasets.train_x,
        y=datasets.train_y,
        validation_data=(datasets.val_x, datasets.val_y),
        epochs=hp.num_epochs,
        batch_size=None,
        callbacks=callback_list,
    )

def test(model, datasets):
    """ Testing routine. Also returns accuracy """

    # Run model on test set
    model.evaluate(
        x=datasets.test_x,
        y=datasets.test_y,
        verbose=1,
    )
    
    # calculate prediction accuracy
    prediction = model.predict(datasets.test_x)
    correct = np.sum(prediction == datasets.test_y)
    return correct / len(prediction)


def main():
    """ Main function. """

    # Map datasets to its path
    datasets_path_dict = {
                            'fer': 'C:\\Users\\kycqi\\Documents\\Brown\\Spring 2020\\cs143\\final\\KCRuWu-emotion-recognition\\data\\fer2013.csv'
                            # 'fer': os.getcwd() + '\\..\\data\\fer.csv'
                        }

    datasets = Datasets(datasets_path_dict[ARGS.data], ARGS.data)

    model = Model()
    
    # Different model input size depending on the dataset. Default is fer2013.
    if ARGS.data is 'fer':
        model(tf.keras.Input(shape=(hp.img_size, hp.img_size, 1)))

    checkpoint_path = "./your_model_checkpoints/"
    model.summary()

    if ARGS.load_checkpoint is not None:
        model.load_weights(ARGS.load_checkpoint)

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # Compile model graph
    model.compile(
        optimizer=model.optimizer,
        loss=model.loss_fn,
        metrics=["sparse_categorical_accuracy"])

    if not ARGS.evaluate:
        train(model, datasets, checkpoint_path)
    
    accuracy = test(model, datasets)
    print("Accuracy: ", accuracy)

# Make arguments global
ARGS = parse_args()

main()
