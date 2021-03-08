from javalang.tokenizer import JavaTokenizer as BaseJavaTokenizer

class JavaTokenizer(BaseJavaTokenizer):
    """Tokenizer that does not replace unicode escape sequences"""

    def pre_tokenize(self):
        self.data = self.decode_data()
        self.length = len(self.data)


def tokenize(code, ignore_errors=False):
    tokenizer = JavaTokenizer(code, ignore_errors)
    return tokenizer.tokenize()
