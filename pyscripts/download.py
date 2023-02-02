#!/usr/bin/python

import requests
import os,sys
import threading
from tqdm import tqdm
import argparse


def download_file(url, path, chunk_size=1024):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("Content-Length", 0))
    progress = tqdm(total=total_size, unit="B", unit_scale=True)

    filename = os.path.join(path, url.split("/")[-1])
    with open(filename, "wb") as f:
        for data in response.iter_content(chunk_size):
            f.write(data)
            progress.update(chunk_size)

    progress.close()
    print(f"Downloaded {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download files from a list of URLs')
    parser.add_argument('urls', metavar='URL', type=str, nargs='+')
    parser.add_argument('--path', type=str, default='.', help='Path to save the downloaded files')

    args = parser.parse_args()

    # create a list of threads
    threads = []
    for url in args.urls:
        t = threading.Thread(target=download_file, args=(url,args.path))
        t.start()
        threads.append(t)

    # wait for all threads to finish
    for t in threads:
        t.join()
