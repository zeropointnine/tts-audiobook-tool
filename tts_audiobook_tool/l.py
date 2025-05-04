from __future__ import annotations
import os
import inspect
import logging
import tempfile
from datetime import datetime
from logging import Logger
from typing import Callable

"""
Logger wrapper
"""
class L:

    logger: Logger

    NORMAL_COLOR: str = ""
    WARNING_COLOR: str = ""
    ERROR_COLOR: str = ""
    MAX_CHARS: int = 100000

    @staticmethod
    def init(name: str, path: str="", level=logging.DEBUG, ansi_colors: bool=False) -> None:
        """ Must be called first """
        if not path:
            path = os.path.join(tempfile.gettempdir(), f"log.log")
        L.logger = logging.getLogger(name)
        logging.basicConfig(filename=path, encoding='utf-8', level=level)

    @staticmethod
    def d(message: str = "") -> None:
        L._go(fn=L.logger.debug, message=message, color_code=L.NORMAL_COLOR)

    @staticmethod
    def i(message: str = "") -> None:
        L._go(fn=L.logger.info, message=message, color_code=L.NORMAL_COLOR)

    @staticmethod
    def w(message: str = "") -> None:
        L._go(fn=L.logger.warning, message=message, color_code=L.WARNING_COLOR)

    @staticmethod
    def e(message: str = "") -> None:
        L._go(fn=L.logger.error, message=message, color_code=L.ERROR_COLOR)

    @staticmethod
    def _go(fn: Callable, message: str = "", color_code: str="") -> None:
        if not L.logger:
            raise Exception("Must first call init()")

        caller_frame = inspect.currentframe().f_back.f_back # type: ignore
        if not caller_frame:
            filename = "[?]"
            line_no = "[?]"
            function_name = "[?]"
        else:
            filename = str( os.path.basename(caller_frame.f_code.co_filename) )
            line_no = caller_frame.f_lineno
            function_name = caller_frame.f_code.co_name

        time = datetime.now().strftime("%H:%M:%S:%f")[:-3]

        string = f"{time} [{filename} {line_no}] [{function_name}] {color_code}{message}"

        if -1 < L.MAX_CHARS < len(string):
            string = string[:L.MAX_CHARS] + " [truncated]"

        fn(string)
