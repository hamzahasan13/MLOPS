import sys
import logging

    ## When error is raised, this function is called
def error_message_detail(error, error_detail:sys):
    """
    It retrieves information about the exception and formats it into a detailed error message containing the filename, line number, 
    and error message. Then, it returns this formatted error message.
    """
    _, _, exc_tb = error_detail.exc_info() ## Gives info on which file/line number the exception has occurred.
    file_name = exc_tb.tb_frame.f_code.co_filename ## Gets the filename which has raised the exception.
    line_number = exc_tb.tb_lineno
    error_msg = str(error)
    
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, line_number, error_msg
    )
    return(error_message)
    
class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)
        
    def __str__(self):
        return self.error_message
        