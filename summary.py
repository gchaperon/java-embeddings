import argparse
import json
import tarfile
from collections import namedtuple
from javalang.tokenizer import tokenize, LexerError
from tqdm import tqdm
from multiprocessing import Pool
from functools import reduce
from itertools import islice
from operator import add
import signal

Data = namedtuple(
    "Data",
    ["files", "tokens", "lines", "bad_syntax", "empty", "library_errors"],
    defaults=(1, *[0] * 5),
)


def is_java_file(tarinfo):
    return tarinfo.name.endswith(".java") and tarinfo.isfile()


def process_source_code(tup):
    """Get number of tokens and number of lines"""
    code, fname = tup

    try:
        tokens = list(tokenize(code))
    except LexerError as e:
        return Data(bad_syntax=1)
    except Exception as e:
        return Data(library_errors=1)

    if tokens:
        return Data(tokens=len(tokens), lines=tokens[-1].position.line)
    else:
        return Data(empty=1)


def extract_java_files(tar):
    for tarinfo in tqdm(
        filter(is_java_file, tar), desc="Processing files", unit="files"
    ):
        with tar.extractfile(tarinfo) as file:
            yield file.read(), tarinfo.name


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def summary(tar_path, processes, chunksize):
    """Process every .java file found in tar and extract aggregated info"""
    summary_data = [0] * len(Data._fields)
    try:
        with tarfile.open(tar_path) as tar, Pool(
            processes, initializer=init_worker
        ) as pool:
            for file_data in pool.imap(
                process_source_code,
                extract_java_files(tar),
                chunksize=chunksize,
            ):
                for i, field in enumerate(file_data):
                    summary_data[i] += field
    except KeyboardInterrupt:
        print("Stopping early, summary might be incomplete")

    return Data(*summary_data)._asdict()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="the java-large tar file")
    parser.add_argument(
        "--num-workers",
        default=None,
        help="number of workers for tokenization, defaults to the number of cores",
    )
    parser.add_argument(
        "--chunksize",
        default=100,
        help="number of files in each batch sent to workers, default 100",
    )
    args = parser.parse_args()
    print(json.dumps(summary(args.file, args.num_workers, args.chunksize), indent=2))
