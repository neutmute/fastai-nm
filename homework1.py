"""Fast AI Lesson 1 Homework"""

#%%
from __future__ import division, print_function
import csv
import os
import json
#from glob import glob
import glob
from datetime import datetime
from collections import namedtuple
import numpy as np
import utils
import vgg16
from vgg16 import Vgg16
from PIL import Image

#%%
class PathConfig2(object):
    """Container for path config that we will be reusing"""
    def __init__(self, root, relative_data):
        self.root = root
        self.data = root + relative_data
        self.test = os.path.join(self.data, 'test')
        self.train = os.path.join(self.data, 'train')
        self.valid = os.path.join(self.data, 'valid')
        self.results = os.path.join(self.root, 'results')

    def get_latest_weight(self):
        """Find the most recently saved weights file - if it exists"""
        weight_filter = os.path.join(self.results, "*.h5")
        weight_files = glob.iglob(weight_filter)

        try:
            latest_weight = max(weight_files, key=os.path.getctime)
            return latest_weight
        except ValueError:      # hack to handle empty results folder
            return ''

def get_vgg(path_config, batch_size):
    """Tune the model and return model"""

    print("Getting model for {p}".format(p=path_config.root))

    vgg = Vgg16()

    existing_weight_file = path_config.get_latest_weight()

    if os.path.exists(existing_weight_file):
        print("Loading weights from {wf}".format(wf=existing_weight_file))
        vgg.model.load_weights(existing_weight_file)
    else:
        # Grab a few images at a time for training and validation.
        # NB: They must be in subdirectories named based on their category
        train_batches = vgg.get_batches(path_config.train, batch_size=batch_size, shuffle=False)
        val_batches = vgg.get_batches(path_config.valid, batch_size=batch_size*2, shuffle=False)

        #print("Learning rate = {lr}".format(lr=vgg.model.optimizer.lr))

        vgg.finetune(train_batches)
        timestamp_as_string = datetime.now().strftime("%Y-%m-%d-%H%M%S")

        epoch_count = 10
        last_history = None
        for epoch in range(epoch_count):
            print("Fitting epoch {c}/{of}".format(c=epoch, of=epoch_count))
            last_history = vgg.fit(train_batches, val_batches, nb_epoch=1)

            weights_filename = "{ts}_{epoch}.h5".format(ts=timestamp_as_string, epoch=epoch)
            weights_file_path = os.path.join(path_config.results, weights_filename)
            print("Saving weights to {weights_file}".format(weights_file=weights_file_path))
            vgg.model.save_weights(weights_file_path)

        print(last_history.history)

    return vgg

def write_csv(predictions):
    """Given predictions, write out the kaggle csv"""
    with open('dogs-cats-submission.csv', 'wb') as csvfile:
        rowwriter = csv.writer(csvfile, delimiter=',')
        rowwriter.writerow(['id', 'label'])
        counter = 1
        predicted_labels = predictions[2]
        for prediction in predicted_labels:
            is_dog = prediction == "dogs"
            is_dog_value = "1" if is_dog else "0"
            rowwwriter.writerow([counter, is_dog_value])
            counter = counter + 1

def get_config():
    """Load paths config"""
    root_path = "data\\cats-dogs-redux"
    relative_data = os.path.sep + "sample"
    #relative_data = os.path.sep + "full"
    path_config = PathConfig2(root_path, relative_data)
    return path_config

def get_predictions():
    """Execute!"""

    test_file_count = sum([len(files) for r, d, files in os.walk(path_config.test)])
    print("Testing '{p}' which has {c} files".format(p=path_config.test, c=test_file_count))
    test_batches, predictions = vgg.test(path_config.test, batch_size=batch_size*2)

    return test_batches, predictions

def debug_predictions(test_batches, predictions, config):
    """Diagnostic tools"""
    print(predictions[:5])

    filenames = test_batches.filenames
    print(filenames[:5])

    image_path = os.path.join(config.test, filenames[2])
    print("Opening " + image_path)
    Image.open(image_path)



#%%
# pylint: disable=C0103

# As large as you can, but no larger than 64 is recommended.
# If you have an older or cheaper GPU, you'll run out of memory, so will have to decrease this.
batch_size = 32

reload(utils)
reload(vgg16)
np.set_printoptions(precision=4, linewidth=100)

path_config = get_config()
vgg = get_vgg(path_config, batch_size)
batches, predictions = get_predictions()


#%%
debug_predictions(batches, predictions, path_config)

#%%
write_csv(predictions)
print("Done")

