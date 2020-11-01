import logging
import settings
from logging import handlers
"""
日志设置
"""

# 日志总设置
my_log = logging.getLogger()
my_log.setLevel('DEBUG')

# 日志格式设置
formatter = logging.Formatter(settings.LOG_BASIC_FORMAT, settings.LOG_DATE_FORMAT)

# 控制台输出设置（不打印DEBUG级别的信息）
chlr = logging.StreamHandler()
chlr.setFormatter(formatter)
chlr.setLevel('INFO')

# 文件打印设置（默认使用logger的级别设置，当达到2MB时对日志文件分文件存储）
file_handler = handlers.RotatingFileHandler(filename=settings.LOG_PATH, maxBytes=2097152)
file_handler.setFormatter(formatter)

# 添加设置
my_log.addHandler(chlr)
my_log.addHandler(file_handler)

logging.info('========== 日志设置完成 ==========')
logging.info('log file save at: %s', settings.LOG_PATH)
