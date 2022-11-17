import argparse
import glob
import os
import random
import shutil

import numpy as np

from utils import get_module_logger


def split(source, destination):
    """
    Create three splits from the processed records. The files should be moved to new folders in the
    same directory. This folder should be named train, val and test.

    args:
        - source [str]: source data directory, contains the processed tf records
        - destination [str]: destination data directory, contains 3 sub folders: train / val / test
    """
    # Load the data files paths using glob and shuffle them
    files = glob.glob(f'{source}/*.tfrecord')
    random.shuffle(files)

     # Get the size of the list of files and create the split sizes: 80/10/10
    n_files = len(files)
    train_len = n_files*80//100
    val_len = n_files*10//100

    # Split the list of data paths into the 3 sections
    train,val,test = files[:train_len], files[train_len:train_len+val_len],files[train_len+val_len:]
    
    # After creating the splits, we will copy the file (img) to the dedicated folder
    for img in train:
        shutil.copy(img,f'{destination}/train')

    for img in val:
        shutil.copy(img,f'{destination}/val')
    
    for img in test:
        shutil.copy(img,f'{destination}/test')
    


    
if  __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--source', required=True,
                        help='source data directory')
    parser.add_argument('--destination', required=True,
                        help='destination data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.source, args.destination)
