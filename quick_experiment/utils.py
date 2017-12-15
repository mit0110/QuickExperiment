"""Helper functions."""
import pickle
import json
import os


def safe_mkdir(dir_name):
    """Checks if a directory exists, and if it doesn't, creates one."""
    try:
        os.stat(dir_name)
    except OSError:
        os.mkdir(dir_name)


def pickle_to_file(object_, filename):
    with open(filename, 'wb') as file_:
        pickle.dump(object_, file_)


def pickle_from_file(filename):
    with open(filename, 'rb') as file_:
        return pickle.load(file_)


def read_config(config_filename):
    """Reads a json file and returns the object"""
    with open(config_filename) as file_:
        return json.load(file_)


def csv_to_file(dataframe, filename, dirname=None):
    """Writes the dataframe with csv format."""
    if dirname:
        safe_mkdir(dirname)
        filename = os.path.join(dirname, filename)
    with open(filename, 'w') as file_:
        dataframe.to_csv(file_, sep='\t')
