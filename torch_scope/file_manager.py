"""
.. module:: file_manager
    :synopsis: file_manager
.. moduleauthor:: Liyuan Liu
"""
import os
import json
import subprocess
from hashlib import sha256

CACHE_ROOT = os.getenv('TS_CACHE_ROOT', os.path.join(os.path.expanduser("~"), ".ts_cache"))
DATASET_ROOT = os.getenv('TS_DATASET_ROOT', os.path.join(os.path.expanduser("~"), ".ts_dataset"))

def cached_url(url):
    """
    Cached a file to the cache folder.
    Parameters
    ----------
    url: ``str``, required.
        The url or the path.
    """
    if os.path.exists(url):
        return url

    hashtag_url = url

    while True:
        url_bytes = hashtag_url.encode('utf-8')
        url_hash = sha256(url_bytes)
        filename = url_hash.hexdigest()
        full_path = os.path.join(CACHE_ROOT, filename)
        config_path = os.path.join(full_path, 'config.json')
        if os.path.exists(full_path) and os.path.exists(config_path):
            with open(config_path, 'r') as fin:
                config = json.load(fin)
            if config['url'] == url:
                return os.path.join(full_path, 'download.cached')
            else:
                hashtag_url += 'l'
        else:
            os.makedirs(full_path, exist_ok = True)
            file_path = os.path.join(full_path, 'download.cached')
            p = subprocess.Popen(["curl", url, "-o", file_path])
            p.communicate()
            # with open(file_path, 'wb') as fout:
            #     c = pycurl.Curl()
            #     c.setopt(c.URL, url)
            #     c.setopt(c.WRITEDATA, fout)
            #     c.perform()
            #     c.close()
            with open(config_path, 'w') as fout:
                json.dump({'url': url}, fout)
            return file_path
