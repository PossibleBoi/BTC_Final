# exception.py
import sys
import traceback

def error_message_detail(error):
    _, _, exc_tb = sys.exc_info()
    if exc_tb:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_no = exc_tb.tb_lineno
        return f"Error in {file_name} at line {line_no}: {str(error)}"
    return str(error)


class CustomException(Exception):
    def __init__(self, error_message, error_detail=None):
        super().__init__(error_message)

        # If only exception is passed: CustomException(e)
        if error_detail is None:
            error_detail = sys

        # format message
        formatted_message = error_message_detail(error_message)
        self.error_message = formatted_message

    def __str__(self):
        return self.error_message
