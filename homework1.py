"""Fast AI Lesson 1 Homework"""

#%%
from __future__ import division, print_function
import csv
import os
import json
from datetime import datetime
from collections import namedtuple
from matplotlib.pyplot import imshow
import numpy as np
import utils            # don't clean up
from utils import *     # both required
import glob
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
        weight_files = glob.glob(weight_filter)

        try:
            latest_weight = max(weight_files, key=os.path.getctime)
            return latest_weight
        except ValueError:      # hack to handle empty results folder
            return ''

def get_vgg(path_config, batch_size):
    """Tune the model and return model"""

    print("Getting model for {p}".format(p=path_config.data))

    vgg = Vgg16()

    existing_weight_file = path_config.get_latest_weight()

    if os.path.exists(existing_weight_file):
        print("Loading weights from {wf}".format(wf=existing_weight_file))
        vgg.model.load_weights(existing_weight_file)
    else:
        # Grab a few images at a time for training and validation.
        # NB: They must be in subdirectories named based on their category
        train_batches = vgg.get_batches(path_config.train, batch_size=batch_size)
        val_batches = vgg.get_batches(path_config.valid, batch_size=batch_size*2)

        vgg.finetune(train_batches)

        vgg.model.optimizer.lr = 0.01
        print("Learning Rate={lr}, Batch size={bs}".format(lr=vgg.model.optimizer.lr, bs=batch_size))

        timestamp_as_string = datetime.now().strftime("%Y-%m-%d-%H%M%S")

        epoch_count = 10
        epoch_history = []
        for epoch in range(epoch_count):
            print("Fitting epoch {c}/{of}".format(c=epoch, of=epoch_count))
            history = vgg.fit(train_batches, val_batches, nb_epoch=1)

            weights_filename = "{ts}_fit_{epoch}.h5".format(ts=timestamp_as_string, epoch=epoch)
            weights_file_path = os.path.join(path_config.results, weights_filename)
            #print("Saving weights to {weights_file}".format(weights_file=weights_file_path))
            vgg.model.save_weights(weights_file_path)
            epoch_history.append(history.history)

            
        print(epoch_history)

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

def get_config(root, relative):
    """Load paths config"""
    root_path = root
    relative_data = os.path.sep + relative
    path_config = PathConfig2(root_path, relative_data)
    return path_config

def get_predictions():
    """Execute our predictions"""

    test_file_count = sum([len(files) for r, d, files in os.walk(path_config.test)])
    print("Predicting (test) '{p}' which has {c} files. This may take some time...".format(p=path_config.test, c=test_file_count))
    test_batches, predictions = vgg.test(path_config.test, batch_size=batch_size*2)
    print("...done")
    return test_batches, predictions

    
#%%
#### Initialise
%matplotlib inline

relative_data_path = "sample"
#relative_data_path = "full"
path_config = get_config("data\\cats-dogs-redux", relative_data_path)
predictions_array_path = os.path.join(path_config.results, "predictions.array")
filenames_array_path = os.path.join(path_config.results, "filenames.array")
expected_labels_array_path = os.path.join(path_config.results, "expected_labels.array")

#%%
#### PREDIT SECTION
# pylint: disable=C0103

# As large as you can, but no larger than 64 is recommended.
# If you have an older or cheaper GPU, you'll run out of memory, so will have to decrease this.
batch_size = 50

reload(utils)
reload(vgg16)
np.set_printoptions(precision=4, linewidth=100)

vgg = get_vgg(path_config, batch_size)

batches, predictions = get_predictions()

save_array(predictions_array_path, predictions)
save_array(filenames_array_path, batches.filenames)
save_array(expected_labels_array_path, batches.classes)

#%%
#### DEBUG SECTION
def plot_indexes(filename_array, indexes, titles=None):
    plots([image.load_img(os.path.join(path_config.test, filename_array[i])) for i in indexes], titles=titles)
 
def debug_predictions(filenames, predictions, config, expected_labels):
    """Diagnostic tools"""

    cat_predictions = predictions[:,0]

    # if we predict cat, then 1-1 = 0 (cat label)
    # if we predict dog, then 1-0 = 1 (dog label)
    our_labels = np.round(1-cat_predictions)
    
    print("predictions")
    print(predictions[:5])
    
    print("cat predictions")
    print(cat_predictions)

    print("expected_labels")
    print(our_labels)
    
    print("our_labels")
    print(our_labels)
        
    print("filenames")
    print(filenames[:5])

    inspect_count = 4

    #plot_confusion_matrix(cm, val_batches.class_indices)

    # Correct labels
    interesting = np.where(our_labels==expected_labels)[0]
    print("Found %d correct labels" % len(interesting))
    interesting_indexes = permutation(interesting)[:inspect_count]
    plot_indexes(filenames, interesting_indexes, our_predictions[interesting_indexes])

    # Incorrect labels
    interesting = np.where(our_labels!=expected_labels)[0]
    print("Found %d *incorrect* labels" % len(interesting))
    interesting_indexes = permutation(interesting)[:inspect_count]
    plot_indexes(filenames, interesting_indexes, our_predictions[interesting_indexes])

    # The images we most confident were class1, and are actually class1The images we most confident were class1, and are actually class1
    correct_class_n = np.where((our_labels==0) & (our_labels==expected_labels))[0]
    print("Found %d confident correct class1 labels" % len(correct_class_n))
    most_correct_class_n = np.argsort(our_predictions[correct_class_n])[::-1][:inspect_count]
    plot_indexes(correct_class_n[most_correct_class_n], our_predictions[correct_class_n][most_correct_class_n])

    # The images we were most confident were class N, but are actually class M
    incorrect_class_n = np.where((our_labels==0) & (our_labels!=expected_labels))[0]
    print("Found %d incorrect class " % len(incorrect_class_n))
    if len(incorrect_class_n):
        most_incorrect_class_n = np.argsort(our_predictions[incorrect_class_n])[::-1][:inspect_count]
        plot_indexes(incorrect_class_n[most_incorrect_class_n], our_predictions[incorrect_class_n][most_incorrect_class_n])
#

   # image_path = os.path.join(config.test, filenames[0])
   # print("Opening " + image_path)
   # i = Image.open(image_path)
   # k = imshow(np.asarray(i))
   # k.title("Ff")

predictions = load_array(predictions_array_path)
filenames = load_array(filenames_array_path)
expected_labels = load_array(expected_labels_array_path)
print(batches.classes)

debug_predictions(filenames, predictions, path_config, expected_labels)

#%%
write_csv(predictions)
print("Done")

#%%
m = np.fromfunction(lambda i, j: (i +1)* 10 + j + 1, (9, 4), dtype=int)
m[:,3]

