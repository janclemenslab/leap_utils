import urllib.request
import logging
import zipfile
import os
import shutil
import tempfile

logging.basicConfig(level=logging.DEBUG)


test_data_url = 'https://www.dropbox.com/sh/oqmuq6ewfl6gngk/AADaLf0u5zXFSstgIUSmO-Eea?dl=1'
test_data_zip = 'tmp.zip'
temp_dir = tempfile.mkdtemp()


def setup():
    """Download test data."""
    logging.info(f'Creating temporary directory {temp_dir}.')
    path_to_zipfile = os.path.join(temp_dir, test_data_zip)

    logging.info(f'Downloading data from {test_data_url} to {path_to_zipfile}.')
    urllib.request.urlretrieve(test_data_url, path_to_zipfile)

    logging.info(f'Unzipping data in {path_to_zipfile}.')
    with zipfile.ZipFile(path_to_zipfile, 'r') as z:
        z.extractall(temp_dir)
    return temp_dir


def teardown():
    """Rempve all test data."""
    logging.info(f'Deleting {temp_dir}.')
    shutil.rmtree(temp_dir)
