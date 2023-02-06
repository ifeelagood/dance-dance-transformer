#!/usr/bin/python



import os
import pathlib
import argparse
import threading 

from manifest import create_manifest
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



def all(args):
    print("Downloading...")
    download_script(args)

    print("Unpacking...")
    unpack_script(args)

    print("Creating manifest...")
    manifest_script(args)
    
def run(args):
    actions = {
        "all": all,
        "download": download_script,
        "unpack": unpack_script,
        "manifest": manifest_script
    }

    func = actions[args.action]
    
    func(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", type=str, choices=["all", "download", "unpack", "manifest"], help="Action to perform.")
    
    args = parser.parse_args()
    
    run(args)