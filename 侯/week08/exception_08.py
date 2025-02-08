"""
@Project ：cgNLPproject 
@File    ：exception_08.py
@Date    ：2025/1/13 14:24 
"""
class FileIOException(Exception):
    def __init__(self, message, keyword):
        self.message = message
        self.keyword = keyword