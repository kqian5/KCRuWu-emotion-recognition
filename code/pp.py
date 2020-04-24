import numpy as np
import hyperparameters as hp
import csv

class Datasets():
    """ Class for containing the training and test sets as well as
    other useful data-related information. Contains the functions
    for preprocessing.
    """

    def get_fer_data(self, path, shuffle):
        with open(path, 'r') as f:
            csv_reader = csv.reader(f)
            fields = next(csv_reader)
            x, y = [], []
            for row in csv_reader:
                emotion = row[0]
                pixels = np.array(row[1].split(' ')).reshape((hp.img_size, hp.img_size, 1))
                x.append(pixels)
                y.append(emotion)
            num_examples = csv_reader.line_num - 1
            x, y = np.array(x), np.array(y).reshape((num_examples, 1))
            train_x, val_x, test_x = np.split(x, [int(0.8 * num_examples), int(0.9 * num_examples)])
            train_y, val_y, test_y = np.split(y, [int(0.8 * num_examples), int(0.9 * num_examples)])
            return train_x, train_y, val_x, val_y, test_x, test_y


    def __init__(self, data_path, task):

        # Map datasets to its own get_data function
        get_data_dict = {
                            'fer': self.get_fer_data,
                        }
        
        # Get train_x, train_y, val_x, val_y, test_x, test_y
        datasets = get_data_dict[task](data_path, True)
        (train_x, train_y, val_x, val_y, test_x, test_y) = datasets
        print('preprocessing complete')




