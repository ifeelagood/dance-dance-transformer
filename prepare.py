#!/usr/bin/python

import os
import pathlib
import argparse
import threading 

from pyscripts import *

PACKS = {
    "fraxtil": ["https://fra.xtil.net/simfiles/data/tsunamix/III/Tsunamix III [SM5].zip", "https://fra.xtil.net/simfiles/data/arrowarrangements/Fraxtil's Arrow Arrangements [SM5].zip", "https://fra.xtil.net/simfiles/data/beastbeats/Fraxtil's Beast Beats [SM5].zip"],
    "itg": ["https://search.stepmaniaonline.net/link/In The Groove 1.zip",  "https://search.stepmaniaonline.net/link/In The Groove 2.zip"]
}


def prepare(args):
    # create download threads
    print("Downloading...")
    download_threads = []
    for pack, urls in PACKS.items():
        pack_dir = args.data_dir / "raw" / pack
                
        for url in urls:
            t = threading.Thread(target=download, args=(url, pack_dir))
            t.start()
            download_threads.append(t)
            
    # wait for downloads to complete
    for t in download_threads:
        t.join()
        
    # extract and flatten
    print("Extracting...")
    
    for pack in PACKS.keys():
        extract_pack(args.data_dir / "raw" / pack)
        
    # proccess
    print("Processing...")
    packs = list(packs.keys())
    process_packs(packs, data_dir=args.data_dir)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=pathlib.Path, default="data", help="Path to store data.")
    
    args = parser.parse_args()
    
    prepare(args)