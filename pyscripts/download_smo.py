#!/usr/bin/python

import requests
import os,sys
import tqdm
import threading
import argparse

def download_file(filename, path, chunk_size=1024):
    s = requests.Session()

    # s.cookies.update({"ci_session": "gldrbnslkd8889hejtd4mffu8f2h93ph"})
    s.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.132 Safari/537.36",
            "Referer": "https://search.stepmaniaonline.net/packs",
            "Origin": "https://search.stepmaniaonline.net",
        })

    # get redirect url
    url = "https://search.stepmaniaonline.net/link/" + filename
    r = s.get(url, allow_redirects=False)

    # get download url
    location = r.headers["Location"]
    if not location.startswith("https://simfiles"):
        raise ValueError("Invalid url from redirect")

    # get file stats
    
    response = requests.get(location, stream=True)
    total_size = int(response.headers.get("Content-Length", 0))
    progress = tqdm.tqdm(total=total_size, unit="B", unit_scale=True)

    filepath = os.path.join(path, filename)
    with open(filepath, "wb") as f:
        for data in response.iter_content(chunk_size):
            f.write(data)
            progress.update(chunk_size)

    progress.close()
    print(f"Downloaded {filename} from stepmaniaonline.net")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download files from a list of URLs')
    parser.add_argument("files", metavar="FILES", type=str, nargs='+')
    parser.add_argument('--path', type=str, default='.', help='Path to save the downloaded files')

    args = parser.parse_args()

    # create a list of threads
    threads = []
    for url in args.files:
        t = threading.Thread(target=download_file, args=(url,args.path))
        t.start()
        threads.append(t)

    # wait for all threads to finish
    for t in threads:
        t.join()
