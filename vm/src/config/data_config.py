from enum import Enum, auto
from typing import Optional
import numpy as np

class ItemType(Enum):
    # Treasure items
    crystal = 0
    diamond = auto()
    emerald = auto()

    # Landmark items
    coin = auto()
    compass = auto()
    coral = auto()
    fossil = auto()
    key = auto()
    letter = auto()
    shell = auto()
    treasure_box = auto()

class Data:
    def __init__(self):
        self.size: tuple[int, int] = (0, 0)
        self.image: np.ndarray = np.array([])
        self.item_types: list[ItemType] = []
        self.bboxes: list[tuple[float, float, float, float]] = []
        self.annotations: list[tuple[ItemType, float, float, float, float]] = []

    def plot(self) -> None:
        pass