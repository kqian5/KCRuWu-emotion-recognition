import numpy as np
import hyperparameters as hp
import csv

class Datasets():
    """ Class for containing the training and test sets as well as
    other useful data-related information. Contains the functions
    for preprocessing.
    """

    def get_fer_data(self, path, flip_prob):
        # open csv file
        with open(path, 'r') as f:
            csv_reader = csv.reader(f)
            fields = next(csv_reader)
            x, y = [], []
            mean = 0
            std = 0
            # loop through images
            for row in csv_reader:
                emotion = int(row[0])
                pixels = np.array(row[1].split(' ')).astype(np.float).reshape((hp.img_size, hp.img_size, 1))
                # normalize
                pixels /= 255.0
                # calculate mean and std
                mean += np.mean(pixels)
                std += np.std(pixels)
                # augment - horizontal flip
                if np.random.random() < flip_prob:
                    augmented_pixels = np.flip(pixels, axis=1)
                    x.append(augmented_pixels)
                    y.append(emotion)
                x.append(pixels)
                y.append(emotion)
            num_examples = csv_reader.line_num - 1
            mean /= num_examples
            std /= num_examples
            x, y = np.array(x), np.array(y).reshape((-1, 1))
            # standardize
            x = (x - mean) / std
            # split up data into train/val/test
            train_x, val_x, test_x = np.split(x, [int(0.8 * num_examples), int(0.9 * num_examples)])
            train_y, val_y, test_y = np.split(y, [int(0.8 * num_examples), int(0.9 * num_examples)])
            return train_x, train_y, val_x, val_y, test_x, test_y


    def __init__(self, data_path, task):

        # Map datasets to its own get_data function
        get_data_dict = {
                            'fer': self.get_fer_data,
                        }
        
        # Get train_x, train_y, val_x, val_y, test_x, test_y
        datasets = get_data_dict[task](data_path, 0.3)
        (self.train_x, self.train_y, self.val_x, self.val_y, self.test_x, self.test_y) = datasets
        print('preprocessing complete')




