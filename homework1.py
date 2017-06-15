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

#PathConfig = namedtuple("PathConfig", "root data test results train")

#%%
reload(utils)
reload(vgg16)
np.set_printoptions(precision=4, linewidth=100)

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

        latest_weight = max(weight_files, key=os.path.getctime)
        
        if len(latest_weight) == 0:
            return ''

        return latest_weight

def get_path_config(root):
    """Capture config that we will be reusing"""
    config2 = PathConfig2(root, os.path.sep + "sample")
    return config2

def get_vgg(path_config, batch_size):
    """Tune the model and return model"""

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

        print("Fitting")
        vgg.fit(train_batches, val_batches, nb_epoch=1)

        timestamp_as_string = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        weights_filename = path_config.results + "/" + timestamp_as_string + ".h5"
        print("Saving weights to {weights_file}".format(weights_file=weights_filename))
        vgg.model.save_weights(weights_filename)

    #test_imgs, labels = next(test_batches)
    #predictions = vgg.predict(test_imgs, True)

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




#%%
def run():
    """Execute!"""

    # As large as you can, but no larger than 64 is recommended.
    # If you have an older or cheaper GPU, you'll run out of memory, so will have to decrease this.
    batch_size = 32

    root_path = "data\\cats-dogs-redux"
    path_config = get_path_config(root_path)

    print("Getting model")
    vgg = get_vgg(path_config, batch_size)

    test_files = os.walk(path_config.test).next()
    print("Testing {p} which has {c} files".format(p=path_config.test, c=len(test_files[2])))
    test_batches, predictions = vgg.test(path_config.test, batch_size=batch_size*2)

    predictions[:5]

run()
print("Done")

#%%
write_csv(predictions)
print("Done")

