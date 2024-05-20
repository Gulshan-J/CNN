
from torch import nn

class loss:
    """
        This class implements the selection of loss function .
    """
    def __new__(cls , name : str) -> str:
        return getattr(cls,name)
    """ This magic method is used for the selection of the loss function.

        Args:
            name (str): name of the loss function.

        Returns:
            str: returns the name of the loss function found in the class.
    """
    def CrossEntropyLoss(self):
        return nn.CrossEntropyLoss()
        
    def BCELoss(self):
        return nn.BCELoss()

