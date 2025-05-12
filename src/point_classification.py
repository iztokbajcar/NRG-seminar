from enum import Enum


class EVodeClassification(Enum):
    """The classification used in the E-vode project."""

    CREATED_UNCLASSIFIED = 0
    UNCLASSIFIED = 1
    GROUND = 2
    LOW_VEGETATION = 3
    MEDIUM_VEGETATION = 4
    HIGH_VEGETATION = 5
    BUILDING = 6
    LOW_POINT = 7
