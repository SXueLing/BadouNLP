import logging
import os

# 创建日志记录器（logger）
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # 设置日志级别为 INFO

# 创建格式器（formatter）
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 创建控制台处理器（Handler1: StreamHandler）
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # 设置控制台日志级别为 INFO
console_handler.setFormatter(formatter)  # 设置格式器

# 创建文件处理器（Handler2: FileHandler）
current_dir = os.path.dirname(os.path.abspath(__file__))
# 动态创建日志路径
log_dir = os.path.join(current_dir, '../logs')
os.makedirs(log_dir, exist_ok=True)  # 如果目录不存在则创建
# 日志文件的路径
log_file = os.path.join(log_dir, 'test.log')
file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')  # 追加模式
file_handler.setLevel(logging.INFO)  # 设置文件日志级别为 INFO
file_handler.setFormatter(formatter)  # 设置格式器

# 将两个处理器添加到日志记录器
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# 输出日志
# logger.info("This is an info message.")
# logger.warning("This is a warning message.")
# logger.error("This is an error message.")
