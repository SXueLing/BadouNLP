import logging
import logging.config

def getLogger(name,config):
    config_dict = config['log_config']
    logging.config.dictConfig(config_dict)
    return logging.getLogger(name)
