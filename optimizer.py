from torch import optim

class optimizer:
    """
        This class implements the selection of optimizer.
    """
    def __new__(cls , name : str) -> str:
        """ This magic method is used for the selection of the optimizer function.

        Args:
            name (str): name of the optimizer function.

        Returns:
            str: returns the name of the optimizer function found in the class.
        """
        return getattr(cls,name)
    
    def adam(self,l_r : float, parameter : any) -> any:
        return optim.Adam(parameter.parameters(), lr = l_r)
    def SGD(self,l_r : float, parameter : any) -> any:
        return optim.SGD(parameter.parameters(), lr = l_r, momentum = 0.09)
