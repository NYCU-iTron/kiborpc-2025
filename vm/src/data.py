from dataclasses import dataclass
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

@dataclass
class Data:
    size: tuple[int, int]
    image = np.ndarray
    annotations: Optional[str] = None
