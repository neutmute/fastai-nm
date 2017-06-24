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

def get_predictions(path):
    """Execute our predictions"""

    file_count = sum([len(files) for r, d, files in os.walk(path)])
    print("Predicting (vgg.test) '{p}' which has {c} files. This may take some time...".format(p=path, c=file_count))
    batches, predictions = vgg.test(path, batch_size=batch_size*2)
    print("...done")
    return batches, predictions

    
#%%
#### Initialise
%matplotlib inline

relative_data_path = "sample"
#relative_data_path = "full"
path_config = get_config("data\\cats-dogs-redux", relative_data_path)
predictions_array_path = os.path.join(path_config.results, "predictions.array")
filenames_array_path = os.path.join(path_config.results, "batches.filenames.array")
expected_labels_array_path = os.path.join(path_config.results, "batches.expected_labels.array")
class_indicies_array_path = os.path.join(path_config.results, "batches.class_indicies.array")

prediction_root = path_config.valid

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

batches, predictions = get_predictions(prediction_root)

save_array(predictions_array_path, predictions)
save_array(filenames_array_path, batches.filenames)
save_array(expected_labels_array_path, batches.classes)
save_array(class_indicies_array_path, batches.class_indices)


#%%
#### DEBUG SECTION
def plot_indexes(folder_root, filename_array, indexes, titles=None):
    plots([image.load_img(os.path.join(folder_root, filename_array[i])) for i in indexes], titles=titles)
 
def debug_predictions(filename_root, filenames, expected_labels, class_indicies, predictions, config):
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

    cm = confusion_matrix(expected_labels, our_labels)
    plot_confusion_matrix(cm, class_indicies)

    our_predictions = cat_predictions
    # Correct labels
    interesting = np.where(our_labels==expected_labels)[0]
    print("Found {good}/{total} correct labels".format(good=len(interesting), total=len(our_labels)))
    interesting_indexes = permutation(interesting)[:inspect_count]
    plot_indexes(filename_root, filenames, interesting_indexes, cat_predictions[interesting_indexes])

    # Incorrect labels
    interesting = np.where(our_labels!=expected_labels)[0]
    print("Found {bad}/{total} *incorrect* labels".format(bad=len(interesting), total=len(our_labels)))
    if (len(interesting) > 0):
        interesting_indexes = permutation(interesting)[:inspect_count]
        plot_indexes(filename_root, filenames, interesting_indexes, cat_predictions[interesting_indexes])

        # The images we most confident were class1, and are actually class1
        correct_class_n = np.where((our_labels==0) & (our_labels==expected_labels))[0]
        #print("Found %d confident correct class1 labels" % len(correct_class_n))
        #print(correct_class_n)
        #print(our_predictions)
        #print(our_predictions[correct_class_n])
        most_correct_class_n = np.argsort(our_predictions[correct_class_n])[::-1][:inspect_count]
        #print(most_correct_class_n)
        #print(correct_class_n[most_correct_class_n])
        #print(our_predictions[correct_class_n][most_correct_class_n])
        plot_indexes(filename_root, correct_class_n[most_correct_class_n], our_predictions[correct_class_n][most_correct_class_n])

    # The images we were most confident were class N, but are actually class M
    incorrect_something = np.where((our_labels==0) & (our_labels!=expected_labels))
    if (len(incorrect_something) > 0):
        incorrect_class_n = incorrect_something[0]
        print("Found %d incorrect class_n " % len(incorrect_class_n))
    # if incorrect_class_n > 0:
    #     most_incorrect_class_n = np.argsort(our_predictions[incorrect_class_n])[::-1][:inspect_count]
    #     plot_indexes(filename_root, incorrect_class_n[most_incorrect_class_n], our_predictions[incorrect_class_n][most_incorrect_class_n])
#

   # image_path = os.path.join(config.test, filenames[0])
   # print("Opening " + image_path)
   # i = Image.open(image_path)
   # k = imshow(np.asarray(i))
   # k.title("Ff")

predictions = load_array(predictions_array_path)
filenames = load_array(filenames_array_path)
expected_labels = load_array(expected_labels_array_path)
class_indicies = load_array(class_indicies_array_path)

debug_predictions(prediction_root, filenames, expected_labels, class_indicies, predictions, path_config)

#%%
write_csv(predictions)
print("Done")

#%%
#slice demo
m = np.fromfunction(lambda i, j: (i +1)* 10 + j + 1, (9, 4), dtype=int)
m[:,3]

