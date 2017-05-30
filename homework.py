"""Fast AI Lesson 1 Homework"""

from __future__ import division, print_function
import csv
import os
import json
from glob import glob
import numpy as np
import utils
import vgg16
from vgg16 import Vgg16

reload(utils)
reload(vgg16)

np.set_printoptions(precision=4, linewidth=100)

def get_predictions():
    """Tune the model and return predictions"""

    path = "data/cats-dogs-redux/sample/"


    # As large as you can, but no larger than 64 is recommended.
    # If you have an older or cheaper GPU, you'll run out of memory, so will have to decrease this.
    batch_size = 32

    vgg = Vgg16()

    # Grab a few images at a time for training and validation.
    # NB: They must be in subdirectories named based on their category
    batches = vgg.get_batches(path+'train', batch_size=batch_size)
    val_batches = vgg.get_batches(path+'valid', batch_size=batch_size*2)
    vgg.finetune(batches)
    vgg.fit(batches, val_batches, nb_epoch=3)

    imgs, labels = next(batches)

    predictions = vgg.predict(imgs, True)

    return predictions

def write_csv(predictions):
    """Given predictions, write out the kaggle csv"""
    with open('dogs-cats-submission.csv', 'wb') as csvfile:
        rowwriter = csv.writer(csvfile, delimiter=',')
        rowwriter.writerow(['id', 'label'])

        rowwwriter.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])

write_csv(get_predictions())


