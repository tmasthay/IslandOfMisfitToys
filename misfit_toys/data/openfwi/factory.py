import os
from misfit_toys.data.dataset import DataFactory
import torch
import numpy as np
import torch


from misfit_toys.swiffer import sco
from misfit_toys.data.dataset import DataFactory

# from misfit_toys.utils import get_pydict

import requests
from googleapiclient.discovery import build
from tqdm import tqdm


def convert_size(num, unit):
    num = float(num)
    base = 1024 if unit.endswith('iB') else 1000
    prefix = unit.replace('iB', 'B').replace('B', '').upper()
    prot = {
        '': num,
        'K': num * base,
        'M': num * base**2,
        'G': num * base**3,
        'T': num * base**4,
        'P': num * base**5,
        'E': num * base**6,
        'Z': num * base**7,
    }
    return prot[prefix]


def download_public_file(
    *,
    file_id,
    file_name,
    api_key,
    dest='',
    chunk_size=1024,
    file_size=0.0,
    static_file_size=False,
):
    """
    Download a public file from Google Drive.

    :param file_id: ID of the file to download.
    :param file_name: Name of the file (for saving locally).
    :param api_key: Your Google API key.
    """
    base_url = f"https://www.googleapis.com/drive/v3/files/{file_id}"
    params = {"key": api_key, "alt": "media"}
    file_path = os.path.join(dest, file_name)
    os.makedirs(dest, exist_ok=True)
    print(f'Requesting {base_url}...', end='')
    response = requests.get(base_url, params=params, stream=True)

    if response.status_code == 200:
        print('SUCCESS', end='\r')
        with open(file_name, 'wb') as f:
            if static_file_size:
                total_size = file_size
            else:
                total_size = int(
                    response.headers.get('content-length', file_size)
                )

            with open(file_path, 'wb') as f, tqdm(
                desc=file_name,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1000,
                bar_format=(dest + '/{l_bar}{bar:10}{r_bar}{bar:-10b}'),
            ) as bar:
                for chunk in response.iter_content(chunk_size):
                    size = f.write(chunk)
                    bar.update(size)
    else:
        print(
            f"FAIL...HTTP status code: {response.status_code}:"
            f" {response.reason}"
        )


def list_and_download_public_files(
    *,
    api_key,
    folder_id,
    num_files,
    dest='',
    chunk_size=1024,
    file_size=0.0,
    static_file_size=False,
):
    """
    List and download files in a public Google Drive folder.

    :param api_key: Your Google API key.
    :param folder_id: The ID of the public Google Drive folder.
    :param num_files: Number of files to list in one request.
    """
    service = build('drive', 'v3', developerKey=api_key)

    results = (
        service.files()
        .list(
            q=f"'{folder_id}' in parents",
            pageSize=num_files,
            fields="nextPageToken, files(id, name)",
        )
        .execute()
    )

    items = results.get('files', [])
    if not items:
        print('No files found.')
    else:
        for item in items:
            download_public_file(
                file_id=item['id'],
                file_name=item['name'],
                dest=dest,
                api_key=api_key,
                file_size=file_size,
                static_file_size=static_file_size,
                chunk_size=chunk_size,
            )


class Factory(DataFactory):
    # def __extend_init__(self):
    #     # data_urls = get_pydict(self.src_path, filename="data_urls")
    #     # model_urls = get_pydict(self.src_path, filename="model_urls")
    #     # self.metadata['data_urls'] = data_urls
    #     # self.metadata['model_urls'] = model_urls
    #     input(self.metadata)

    def _manufacture_data(self):
        print(self.src_path)
        self.download_all()

    def download_instance(self, k, indices='all'):
        prev_res = [e for e in os.listdir(self.out_path) if e.endswith('.npy')]
        for f in numpy_files:
            self.tensors[f.replace('.npy', '')] = torch.from_numpy(
                np.load(f)
            ).permute(0, 1, 3, 2)
            try:
                os.remove(f)
            except PermissionError:
                print(f'PermissionError: removal of {f}')


class FactorySignalOnly(DataFactory):
    def _manufacture_data(self):
        pass


def signal_children():
    factory = FactorySignalOnly.cli_construct(
        device="cuda:0", src_path=os.path.dirname(__file__)
    )
    factory.manufacture_data()


if __name__ == "__main__":
    signal_children()
