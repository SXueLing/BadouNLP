
class FileIOException(Exception):
    def __init__(self, message, keyword):
        self.message = message
        self.keyword = keyword