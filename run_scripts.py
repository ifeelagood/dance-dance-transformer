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
        pack_dir = args.data_dir / "raw" / pack
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
        for f in pack_dir.iterdir():
            t = threading.Thread(target=unzip, args=(f, pack_dir))
            t.start()
            extract_threads.append(t)

    # wait for extractions to complete
    for t in extract_threads:
        t.join()

def process_script(args):
    process_packs(config)

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
    
    args = parser.parse_args()
    
    run(args)