"""
Assumes (has been performed manually)
 root folder like 'state-farm'
 full data in 'state-farm/full'
 train files reside in 'test/unknown' folder

Intent:
* Ensure results folder exists
* Create sample folder and sub paths (Train, Valid, Test)
* Create full/valid folder
* Populate sample folder and sub paths (Train, Valid, Test)
* Populate full/valid folder
"""

from glob import glob
from random import shuffle
import os
import shutil

class PathConfig(object):
    """Container for path config that we will be reusing"""
    def __init__(self, root, relative_data):
        self.root = root
        self.data = root + relative_data
        self.test = os.path.join(self.data, 'test')
        self.train = os.path.join(self.data, 'train')
        self.valid = os.path.join(self.data, 'valid')
        self.results = os.path.join(self.root, 'results')

def ensure_folder_exists(path):
    """Create a folder if it doesn't exist"""
    if not os.path.exists(path):
        os.mkdir(path)

def populate_valid_folder(path_config, percent=0.8):
    """Populate the validation folder"""
    print "Testing if we need to do work"
    files_in_valid_folder = glob(os.path.join(path_config.valid, '*.*'))
    if len(files_in_valid_folder) > 2:
        print "Already split. Not populating valid folder"
        return
    print "Getting our split"
    train_set, valid_set = get_split_set(path_config.train, percent)
    for training_file in train_set:
        shutil.move(training_file, train_path)


def sample_folder(path_config_full, path_config_sample, percent=0.95):
    """ create sample folder """
    train_path = os.path.join(path, 'train')
    all_train = glob(os.path.join(train_path, '*.*'))
    sample_path = os.path.join(path, 'sample')
    if not os.path.exists(sample_path):
        os.mkdir(sample_path)
    train_set, sample_set = get_split_set(all_train, percent)
    for sample_file in sample_set:
        shutil.copy(sample_file, sample_path)

def get_split_set(folder, percent):
    """"Returns two dimensional array of files before split, files after split"""
    shuffle(folder)
    train_file_count = len(folder)
    split_point = int(percent * train_file_count)
    return folder[:split_point], folder[split_point:]

def main():
    """Setup our folders"""
    root_path = "data\\state-farm"

    sample_data_path = "\\sample"
    full_data_path = "\\full"
    path_config_sample = PathConfig(root_path, sample_data_path)
    path_config_full = PathConfig(root_path, full_data_path)

    print "Ensure results folder exists"
    ensure_folder_exists(path_config_sample.results)

    print "Create sample folder and sub paths (Train, Valid, Test)"
    ensure_folder_exists(path_config_sample.data)
    ensure_folder_exists(path_config_sample.train)
    ensure_folder_exists(path_config_sample.valid)
    ensure_folder_exists(path_config_sample.test)

    print "Create full/valid folder"
    ensure_folder_exists(path_config_full.valid)

    print "Populating full/valid folder"
    # https://stackoverflow.com/questions/13610074/is-there-a-rule-of-thumb-for-how-to-divide-a-dataset-into-training-and-validatio
    populate_valid_folder(path_config_full, 0.8)
    print "Done"

#    train_valid_percent_split = 0.9
#    train_sample_percent_split = 0.95   
#    #    
#    # Main train, valid Folders 
#    path = parent_path  
#    sample_folder(path,train_sample_percent_split)
#    train_valid_folders(path,train_valid_percent_split)
#    
#

if __name__ == '__main__':
    main()
