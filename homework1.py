"""Fast AI Lesson 1 Homework"""

#%%
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

#%%
def get_predictions():
    """Tune the model and return predictions"""

    #path = "data/cats-dogs-redux/train/"
    path = "data/cats-dogs-redux/sample/"

    # As large as you can, but no larger than 64 is recommended.
    # If you have an older or cheaper GPU, you'll run out of memory, so will have to decrease this.
    batch_size = 32

    vgg = Vgg16()

    # Grab a few images at a time for training and validation.
    # NB: They must be in subdirectories named based on their category
    train_batches = vgg.get_batches(path + 'train', batch_size=batch_size, shuffle=False)
    val_batches = vgg.get_batches(path + 'valid', batch_size=batch_size*2, shuffle=False)

    #print("Learning rate = {lr}" % {lr:vgg.model.optimizer.lr})

    vgg.finetune(train_batches)

    print("Fitting")
    vgg.fit(train_batches, val_batches, nb_epoch=1)

    #test_imgs, labels = next(test_batches)

    predictions = vgg.predict(test_imgs, True)

    return predictions

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
print("Getting predictions")
predictions = get_predictions()
#%%
write_csv(predictions)
print("Done")

