import sys,logging

#from loguru import logger

def __exception_handle(e_type, e_value, e_traceback):
    if issubclass(e_type, KeyboardInterrupt):
        print('\nBye...')
        sys.exit(0)

    sys.__excepthook__(e_type, e_value, e_traceback)

#class __InterceptHandler(logging.Handler):
#    def emit(self, record):
#        try:
#            level = logger.level(record.levelname).name
#        except ValueError:
#            level = record.levelno
#
#        frame, depth = logging.currentframe(), 2
#        while frame.f_code.co_filename == logging.__file__:
#            frame = frame.f_back
#            depth += 1

#        logger.opt(depth=depth, exception=record.exc_info).log(
#            level, record.getMessage()
#        )

def hook_except_handle():
    sys.excepthook = __exception_handle

#def hook_logging(**kwargs):
#    logging.basicConfig(handlers=[__InterceptHandler()], **kwargs)
