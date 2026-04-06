from abc import ABC, abstractmethod
from typing import Any

# game interface
class Game_Interface(ABC):

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def transition(self, state, action) -> Any:
        pass