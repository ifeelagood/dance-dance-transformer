#!/usr/bin/python



import os
import shutil
import pathlib
import argparse
import threading 

from manifest import create_manifest

from dataset import train_valid_split, generate_shards, delete_shards, analyse_dataset
from pyscripts import *
from config import config

def download_script(args):
    if not config.paths.raw.exists():
        config.paths.raw.mkdir(parents=True)

    download_threads = []
    for pack, urls in config.dataset.urls.items():
        pack_dir = config.paths.raw / pack

        # create pack dir
        if not pack_dir.exists():
            pack_dir.mkdir(parents=True)

        for url in urls:
            t = threading.Thread(target=download, args=(url, pack_dir))
            t.start()
            download_threads.append(t)

    # wait for downloads to complete
    for t in download_threads:
        t.join()


def unpack_script(args):
    extract_threads = []
    for pack in config.dataset.packs:
        pack_dir = config.paths.raw / pack
        t = threading.Thread(target=unpack, args=(pack_dir,))
        t.start()
        extract_threads.append(t)

    # wait for extractions to complete
    for t in extract_threads:
        t.join()

def manifest_script(args):
    create_manifest(config)

def analyse_script(args):
    analyse_dataset()

def generate_script(args):
    # create webdataset path
    if not config.paths.shards.exists():
        config.paths.shards.mkdir(parents=True, exist_ok=True)
    elif len(os.listdir(config.paths.shards)):
        delete_shards()


    # get manifest
    train_manifest, valid_manifest = train_valid_split()

    # create threads
    t1 = threading.Thread(target=generate_shards, args=(train_manifest, "train"))
    t2 = threading.Thread(target=generate_shards, args=(valid_manifest, "valid"))

    # start threads
    t1.start()
    t2.start()

    # wait for threads()
    t1.join()
    t2.join()


def all(args):
    print("Downloading...")
    download_script(args)

    print("Unpacking...")
    unpack_script(args)

    print("Creating manifest...")
    manifest_script(args)

    print("Generating dataset")
    generate_script(args)
    
def run(args):
    actions = {
        "all": all,
        "download": download_script,
        "unpack": unpack_script,
        "manifest": manifest_script,
        "analyse": analyse_script,
        "generate": generate_script
    }

    func = actions[args.action]
    
    func(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", type=str, choices=["all", "download", "unpack", "manifest", "analyse", "generate"], help="Action to perform.")
    
    args = parser.parse_args()
    
    run(args)