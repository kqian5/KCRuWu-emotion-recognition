import os
import argparse
import numpy as np
import tensorflow as tf
from model import Model
import hyperparameters as hp
from preprocess import Datasets
from tensorboard_utils import ImageLabelingLogger, ConfusionMatrixLogger
import matplotlib.pyplot as plt
import cv2

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
    parser.add_argument(
        '--live',
        action='store_true',
        help='''Live webcam feedback''')

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
    model.fit(
        x=datasets.train_x,
        y=datasets.train_y,
        validation_data=(datasets.val_x, datasets.val_y),
        epochs=hp.num_epochs,
        batch_size=64,
        callbacks=callback_list,
    )

def test(model, datasets):
    """ Testing routine. Also returns accuracy """

    # Run model on test set
    print('\n# Evaluate on test data')
    results = model.evaluate(
        x=datasets.test_x,
        y=datasets.test_y,
        verbose=1,
    )
    print('test loss, test acc:', results)

def prediction_visualization(model, datasets):
    
    indices = np.random.randint(low=0, high=len(datasets.test_x)-1, size=10)
    inputs = datasets.test_x[indices]
    
    probs = model.predict(
        x=inputs
    )
    predictions = np.argmax(probs, axis=1)
    labels = datasets.test_y[indices].flatten()
    print(predictions)
    print(labels)
    print(np.sum(labels == predictions))
    
    visualize(inputs, predictions, labels)

    id_to_emotion = {
        0:'Angry',
        1:'Disgust',
        2:'Fear',
        3:'Happy',
        4:'Sad',
        5:'Surprise',
        6:'Neutral'
    }

    fig = plt.figure()
    for i in range(1,11):
        plot = fig.add_subplot(2, 5, i)
        title = 'True Label = ' + id_to_emotion[labels[i-1]] + ' \n Prediction = ' + id_to_emotion[predictions[i-1]]
        plot.title.set_text(title)
        plt.imshow(inputs[i-1].reshape((48,48)), cmap='binary_r', vmin=0, vmax=255)
    plt.show()

def live(model):
    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #480 x 640

        square = gray[:, 80:560]

        downsample = cv2.resize(square, (48,48), interpolation = cv2.INTER_AREA)
        
        # print(downsample.shape)
        # display = np.concatenate((gray, downsample), axis=1)
        # Display the resulting frame

        id_to_emotion = {
            0:'Angry',
            1:'Disgust',
            2:'Fear',
            3:'Happy',
            4:'Sad',
            5:'Surprise',
            6:'Neutral'
        }

        probs = model.predict(
            x=downsample.reshape((1,48,48,1)).astype(np.float)
        )
        prediction = np.argmax(probs, axis=1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(gray,  
                id_to_emotion[prediction[0]],  
                (50, 50),  
                font, 1,  
                (0, 255, 255),  
                2,  
                cv2.LINE_4)

        cv2.imshow('frame', gray)

        if cv2.waitKey(2) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def main():
    """ Main function. """

    # Map datasets to its path
    datasets_path_dict = {
                            'fer': os.getcwd() + '/../data/fer2013.csv'
                        }

    datasets = Datasets(datasets_path_dict[ARGS.data], ARGS.data)

    model = Model()
    
    # Different model input size depending on the dataset. Default is fer2013.
    if ARGS.data == 'fer':
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
    
    if ARGS.live:
        live(model)
    test(model, datasets)

    # prediction_visualization(model, datasets)

# Make arguments global
ARGS = parse_args()

main()
