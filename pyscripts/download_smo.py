import requests
import os,sys
import tqdm
import threading
import argparse

def get_url_smo(url):
    """Retrieve the download link from stempaniaonline.net"""
    s = requests.Session()

    s.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.132 Safari/537.36",
            "Referer": "https://search.stepmaniaonline.net/packs",
            "Origin": "https://search.stepmaniaonline.net",
        })

    # get redirect url
    r = s.get(url, allow_redirects=False)

    # get download url
    location = r.headers["Location"]
    if not location.startswith("https://simfiles"):
        raise ValueError("Invalid url from redirect")

    return location