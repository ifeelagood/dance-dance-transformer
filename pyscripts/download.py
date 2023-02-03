import requests
import os
from urllib.parse import urlparse, unquote
from tqdm import tqdm


def url2filename(url):
    return os.path.basename(unquote(urlparse(url).path))

def download_file(url, path, chunk_size=1024):
    """Download a file from the given url to the given path"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("Content-Length", 0))
    progress = tqdm(total=total_size, unit="B", unit_scale=True)

    filename = os.path.join(path, url2filename(url))
    with open(filename, "wb") as f:
        for data in response.iter_content(chunk_size):
            f.write(data)
            progress.update(chunk_size)

    progress.close()

def get_url_smo(url):
    """Retrieve the download link for stempaniaonline.net"""
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

def download(url, path, chunk_size=1024):
    if url.startswith("https://search.stepmaniaonline.net"):
        url = get_url_smo(url)
        
    download_file(url, path, chunk_size)
