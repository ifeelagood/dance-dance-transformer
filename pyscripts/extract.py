#!/usr/bin/python

import os
import zipfile
import argparse
import shutil

def is_simfile(path):
    """Returns true if the path is a simfile"""
    return path.endswith(".sm") or path.endswith(".ssc")

def is_song(path):
    return any(is_simfile(f) for f in os.listdir(path))

def extract_pack(path):
    """Extracts contents of all zip files to the top level directory"""

    # get all zip files in the directory
    zip_files = [f for f in os.listdir(path) if f.endswith(".zip")]

    for zip_file in zip_files:
        # get the path to the zip file
        zip_path = os.path.join(path, zip_file)

        # extract the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(path)

        # delete the zip file
        os.remove(zip_path)

    # recursively flatten directories, leaving only song dirs
    for root, dirs, files in os.walk(path):
        for d in dirs:
            # get the path to the directory
            dir_path = os.path.join(root, d)

            # if the directory is a song, move the directory to the top level
            if is_song(dir_path):
                # if the directory already exists, skip it
                if os.path.exists(os.path.join(path, d)):
                    print(f"WARNING: {d} already exists")
                    continue

                # move the directory
                os.rename(dir_path, os.path.join(path, d))

    # remove all top level directories that are not songs
    for d in os.listdir(path):
        # get the path to the directory
        dir_path = os.path.join(path, d)

        # if the directory is not a song, remove it
        if not is_song(dir_path):
            shutil.rmtree(dir_path)
