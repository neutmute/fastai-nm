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

#def train_valid_folders(path,percent=0.8,flag=0):
#    train_path = os.path.join(path,'train')
#    valid_path = os.path.join(path,'valid')
#    result_path = os.path.join(path,'results')
#    
#    if not os.path.exists(train_path):
#        os.mkdir(train_path)
#    if not os.path.exists(valid_path):
#        os.mkdir(valid_path)    
#    if not os.path.exists(result_path):
#        os.mkdir(result_path)
#    
#    if flag:
#        all_train = glob(os.path.join(path,'*.*'))
#    else:
#        all_train = glob(os.path.join(train_path,'*.*'))
#    train_set,valid_set = get_split_set(all_train,percent)
#    for f in valid_set:
#        shutil.move(f,valid_path)
#    if flag:
#        for f in train_set:
#            shutil.move(f,train_path)


def sample_folder(path, percent=0.95):
    """ create sample folder """
    train_path = os.path.join(path, 'train')
    all_train = glob(os.path.join(train_path, '*.*'))
    sample_path = os.path.join(path, 'sample')
    if not os.path.exists(sample_path):
        os.mkdir(sample_path)
    _, sample_set = get_split_set(all_train, percent)
    for sample_file in sample_set:
        shutil.copy(sample_file, sample_path)

def get_split_set(all_train, per):
    """"Something to do with split sets"""
    shuffle(all_train)
    train_file_count = len(all_train)
    split_point = int(per * train_file_count)
    return all_train[:split_point], all_train[split_point:]

def main():
    """Setup our folders"""
    root_path = "data\\state-farm"

    sample_data_path = "sample"
    full_data_path = "full"
    path_config_sample = PathConfig(root_path, sample_data_path)
    path_config_full = PathConfig(root_path, full_data_path)

    print "Ensure results folder exists"
    ensure_folder_exists(path_config_sample.results)

    print "Create sample folder and sub paths (Train, Valid, Test)"
    ensure_folder_exists(path_config_sample.train)
    ensure_folder_exists(path_config_sample.valid)
    ensure_folder_exists(path_config_sample.test)

    print "Create full/valid folder"
    ensure_folder_exists(path_config_full.valid)

#    parent_path = 'data/cats-dogs-redux'
#    train_zip_path = os.path.join(parent_path,'train.zip')
#    test_zip_path = os.path.join(parent_path,'test.zip')
#    train_valid_percent_split = 0.9
#    train_sample_percent_split = 0.95   
#    
#   #    with zipfile.ZipFile(train_zip_path, "r") as z:
#   #        z.extractall(parent_path)
#   #    with zipfile.ZipFile(test_zip_path, "r") as z:
#   #        z.extractall(parent_path)
#    
#    # Main train, valid Folders 
#    path = parent_path  
#    sample_folder(path,train_sample_percent_split)
#    train_valid_folders(path,train_valid_percent_split)
#    
#    # Cats, Dog folders with train and valid folders of Train folder
#    path = os.path.join(parent_path,'train')
#    create_cats_dogs(path)
#    path = os.path.join(parent_path,'valid')
#    create_cats_dogs(path)
#
#    
#    # Train, Valid folders within Sample Folder
#    path = os.path.join(parent_path,'sample')
#    train_valid_folders(path,train_valid_percent_split,1)   
#    
#    # Cats, Dog folders with train folder of Sample 
#    child_path = os.path.join(parent_path,'sample')
#    train_path = os.path.join(child_path,'train')
#    create_cats_dogs(train_path)
#
#    # Cats, Dog folders with valid folder of Sample
#    child_path = os.path.join(parent_path,'sample')
#    valid_path = os.path.join(child_path,'valid')
#    create_cats_dogs(valid_path)
#

if __name__ == '__main__':
    main()
