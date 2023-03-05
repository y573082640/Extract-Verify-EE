from datetime import datetime
import logging
import os
import sys


def logger_init(log_file_name='monitor',
                log_level=logging.DEBUG,
                log_dir='./log/',
                only_file=False):
    # 指定路径
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_path = os.path.join(log_dir, log_file_name +
                            '_' + str(datetime.now()) + '.txt')
    formatter = '[%(asctime)s] - %(levelname)s: %(message)s'
    if only_file:
        logging.basicConfig(filename=log_path,
                            level=log_level,
                            format=formatter,
                            datefmt='%Y-%d-%m %H:%M:%S')
    else:
        logging.basicConfig(level=log_level,
                            format=formatter,
                            datefmt='%Y-%d-%m %H:%M:%S',
                            handlers=[logging.FileHandler(log_path),
                                      logging.StreamHandler(sys.stdout)]
                            )
