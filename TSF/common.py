
from abc import ABC, abstractmethod

class TSInterface(ABC):
    # Time Series Interfacce for defining the methods that must
    # be implemented by other classes

    @abstractmethod
    def fit(X, y) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def predict():
        raise NotImplementedError