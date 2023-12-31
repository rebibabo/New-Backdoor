# -*- coding: utf-8 -*-

import argparse
import os

import gdown

from ncc import (
    __TREE_SITTER_LIBS_DIR__,
    LOGGER,
)
from ncc.utils.path_manager import PathManager

PathManager.mkdir(__TREE_SITTER_LIBS_DIR__)

TREE_SITTER_SO_FILE_ARCHIVE_MAP = {
    "c": "https://drive.google.com/uc?id=1Ce0Wp_IYw4a69dMAd4RbaOqRK-DD592G",
    "cpp": "https://drive.google.com/uc?id=1Ip-_lW95I7DU_wj96CR-j31VehLtJLz2",
    "csharp": "https://drive.google.com/uc?id=1fCnNd3WiU1aVqgYZ9ygydTgedHq09pzw",
    "go": "https://drive.google.com/uc?id=18nIHKBahzkK4Xgm5mHRCOY2npiTC2NLd",
    "java": "https://drive.google.com/uc?id=1lP-H7D0IpqijmaseigcyqkKBzxWdwmYH",
    "javascript": "https://drive.google.com/uc?id=1OxM0VFhDi2P8WsOuL0pKzZ8MD-CErzqP",
    "julia": "https://drive.google.com/uc?id=13_GehtPCUgD1Df6p1-CF0vcEfzMtBTEj",
    "nix": "https://drive.google.com/uc?id=13W5w4OgcmTEakOSOVGvtqmm97_Px6O5z",
    "php": "https://drive.google.com/uc?id=1lGzi98rQn4qRnidKpn0jchL8QyLS6gUT",
    "python": "https://drive.google.com/uc?id=1jhadgdOng1I95cwtmNJz2fqW-SUvhpch",
    "ruby": "https://drive.google.com/uc?id=1geDqNll4ewd8zqmvUPg9uNrCMZ1iHbQz",
}


def download(name):
    if name in TREE_SITTER_SO_FILE_ARCHIVE_MAP:
        url = TREE_SITTER_SO_FILE_ARCHIVE_MAP[name]
        LOGGER.info(f"Download {name}.so from {url}")
        gdown.download(url=url, output=os.path.join(__TREE_SITTER_LIBS_DIR__, f"{name}.so"))
    else:
        raise FileExistsError(
            f"{name}.so has not been uploaded to the server. Please, build {name}.so with " \
            f" {os.path.dirname(__file__)}/build_so.py"
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Downloading Tree-Sitter Library(ies)")
    parser.add_argument(
        "--names", "-n", type=str, nargs='+', help="TreeSitter language names",
        default=list(TREE_SITTER_SO_FILE_ARCHIVE_MAP.keys()),
    )
    args = parser.parse_args()

    for name in args.names:
        download(name)
