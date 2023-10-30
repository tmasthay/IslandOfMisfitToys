from masthay_helpers.global_helpers import vco, prettify_dict
import os
import gdown
import torch
import numpy as np


def get_hashes(filename):
    base_lines = f"cat {filename} | grep 'Storage used: ' | grep 'tooltip'"

    extract_first_cmd = base_lines + " | ".join(
        [
            "",
            "head -n 1",
            "grep -oP '<c-data.*</c-data>'",
            "awk -F';' '{print $(NF-1)}'",
        ]
    )

    extract_second_cmd = base_lines + " | ".join(
        [
            "",
            """grep -oP 'data-id=".{10,50}" class='""",
            """awk -F'"' '{print $2}'""",
        ]
    )

    u = [vco(extract_first_cmd)]
    u.extend(vco(extract_second_cmd).split('\n'))
    return u


def get_urls(filename):
    hashes = get_hashes(filename)
    return {
        f'data{i+1}': f"https://drive.google.com/uc?id={hash}"
        for i, hash in enumerate(hashes)
    }


def save_urls(filename):
    urls = get_urls(filename)
    path = os.path.dirname(filename)
    s = prettify_dict(urls)
    with open(f'{path}/urls.pydict', 'w') as f:
        f.write(s)


def download_directory(subpath, chooser):
    urls = get_urls(os.path.join(subpath, 'data.html'))
    res = chooser(urls)
    for k, v in res.items():
        base = f'{subpath}/{k}'
        gdown.download(v, f'{base}.npy', quiet=False)
        tensor = torch.from_numpy(np.load(f"{base}.npy"))
        torch.save(tensor, f"{base}.pt")
        os.remove(f"{base}.npy")
