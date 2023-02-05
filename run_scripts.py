#!/usr/bin/python



import os
import pathlib
import argparse
import threading 

from pyscripts import *
from config import config

def download_script(args):
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

def process_script(args):
    # create examples directory
    if not config.paths.examples.exists():
        config.paths.examples.mkdir(parents=True)
    
    preprocess(config, num_threads=args.num_threads)

def all(args):
    print("Downloading...")
    if not config.paths.raw.exists():
        config.paths.raw.mkdir(parents=True)
    download_script(args)

    print("Unpacking...")
    unpack_script(args)

    print("Processing...")
    if not config.paths.wav.exists():
        config.paths.wav.mkdir(parents=True)

    process_script(args)
    
def run(args):
    actions = {
        "all": all,
        "download": download_script,
        "unpack": unpack_script,
        "process": process_script
    }

    func = actions[args.action]
    
    func(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", type=str, choices=["all", "download", "unpack", "process"], help="Action to perform.")
    parser.add_argument("--num-threads", type=int, default=16, help="Number of threads to use when preprocessing.")
    
    args = parser.parse_args()
    
    run(args)