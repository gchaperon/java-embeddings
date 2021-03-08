import os
import tarfile
from javalang.tokenizer import LexerError
from tokenizer import tokenize
from tqdm import tqdm
import argparse
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from itertools import islice
import logging
from gensim.models import Word2Vec

logger = logging.getLogger(__name__)


class JavaLargeCorpus:
    resources = {
        "java-large": "java-large.tar.gz",
        "preprocessed": "java-large-preprocessed.txt",
    }

    def __init__(self, data_path, max_sentences=None):
        """Expects data_path to contain the java-large tar file.
        At most max_sentences will be yielded (None for all)"""

        self.tar_path = Path(data_path) / self.resources["java-large"]
        self.cache_path = Path(data_path) / self.resources["preprocessed"]
        self.max_sentences = max_sentences

    def __iter__(self):
        if self.cache_path.exists():
            logger.info(
                f"Found preprocessed file {self.cache_path}, "
                "skipping preprocessing of java-large"
            )
            yield from islice(self._iter_from_cached(), self.max_sentences)
        else:
            logger.info("Preprocessing corpus, this pass might take longer")
            logger.info(f"Storing preprocessed files to {self.cache_path}")
            yield from islice(self._preprocess(), self.max_sentences)

    def _iter_from_cached(self):
        with open(self.cache_path) as file:
            for line in file:
                yield line.split()

    def _preprocess(self):
        with tarfile.open(self.tar_path) as tar, Pool() as pool, open(
            self.cache_path, "w"
        ) as cache_file:
            code_generator = (
                self.extract_code(tar, tarinfo)
                for tarinfo in filter(self.is_java_file, tar)
            )
            for tokens in pool.imap_unordered(
                self.process_code, code_generator, chunksize=100
            ):
                # The file might empty or have bad syntax, in wich case
                # the tokens list will be empty
                if tokens:
                    cache_file.write(" ".join(tokens) + "\n")
                    yield tokens

    @staticmethod
    def is_java_file(tarinfo):
        return tarinfo.name.endswith(".java") and tarinfo.isfile()

    @staticmethod
    def process_code(code):
        try:
            return [tok.value for tok in tokenize(code)]
        except LexerError:
            return []

    @staticmethod
    def extract_code(tar, tarinfo):
        codecs = ["utf-8", "iso-8859-1"]
        with tar.extractfile(tarinfo) as file:
            for codec in codecs:
                try:
                    return file.read().decode(codec)
                except UnicodeDecodeError:
                    pass
            else:
                raise UnicodeDecodeError(f"Cannot decode file {tarinfo.name}")


def main(data_path):
    sentences = JavaLargeCorpus(data_path)
    model = Word2Vec(sentences=sentences, workers=os.cpu_count())
    model.wv.save_word2vec_format(data_path + "/model.vec")


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s] %(name)s - %(levelname)s: %(message)s", level=logging.INFO
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_path",
        metavar="DATADIR",
        help="the dir where the java-large tar is located, and where the "
        "preprocessed files will be stored",
    )
    args = parser.parse_args()
    main(**vars(args))
