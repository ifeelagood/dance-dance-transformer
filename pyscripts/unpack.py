#!/usr/bin/python

import os
import zipfile
import pathlib
import shutil

def is_simfile(path):
    """Returns true if the path is a simfile"""
    return path.endswith(".sm") or path.endswith(".ssc")

def is_song(path):
    return any(is_simfile(f) for f in os.listdir(path))


def unzip(pack_path, delete=True):
    """Unzip all zip files in the given directory"""

    # find all zip files in the directory
    zip_files = [f for f in os.listdir(pack_path) if f.endswith(".zip")]

    for zip_file in zip_files:
        # get the path to the zip file
        zip_path = os.path.join(pack_path, zip_file)

        # extract the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(pack_path)

        if delete:
            # delete the zip file
            os.remove(zip_path)


def flatten(pack_path, overwrite=False):
    """Recursively move all songs in a directory to the top level directory."""

    for root, dirs, files in os.walk(pack_path):
        for d in dirs: # for each directory:

            dir_path = pathlib.Path(root, d)

            if is_song(dir_path): # if the directory is a song, move the directory to the top level
                target_path = pathlib.Path(pack_path, d)
                # if the directory already exists, replace it.
                if target_path.exists():
                    if overwrite:
                        print(f"WARNING: Overwriting {target_path}")
                        shutil.rmtree(target_path)

                    else:
                        print(f"WARNING: Skipping {target_path}")
                        continue

                # move the directory
                os.rename(dir_path, os.path.join(pack_path, d))

def clean(path):
    """Remove empty non-song empty directories in the given directory."""
    for root, dirs, files in os.walk(path):
        for d in dirs:
            dir_path = pathlib.Path(root, d)

            # if the directory is empty, remove it
            if len(os.listdir(dir_path)) == 0:
                shutil.rmtree(dir_path)

            # if the directory is not a song, remove it
            elif not is_song(dir_path):
                shutil.rmtree(dir_path)
    
            

def unpack(pack_dir, delete=True, overwrite=False):
    """Extract all zip files, flatten the directory, and remove empty directories."""

    # recursively unzip all zip files
    unzip(pack_dir, delete=delete)

    # recursively flatten the directory
    flatten(pack_dir, overwrite=overwrite)

    # recursively remove empty directories
    clean(pack_dir)

