import xml.etree.ElementTree as ET
from typing import *

from hcat.state import Piece, Cell


def data_from_xml(f: str) -> Tuple[List[List[int]], List[int]]:
    """
    Parse a XML data file from labelImg to the raw box annotatiosn

    :param f: xml file
    :return: List -> Boxes, Labels
    """
    root = ET.parse(f).getroot()

    boxes: List[List[int]] = []
    labels: List[int] = []
    for c in root.iter("object"):
        for box, cls in zip(c.iter("bndbox"), c.iter("name")):
            label: str = cls.text

            if label in ["OHC", "IHC"]:
                label: int = 1 if label == "OHC" else 2
                box = [
                    int(box.find(s).text) for s in ["xmin", "ymin", "xmax", "ymax"]
                ]

                boxes.append(box)
                labels.append(label)

    return boxes, labels

def cells_from_xml(f, parent: Piece):
    """ returns a bunch of cells from an xml file"""

    boxes, labels = data_from_xml(f)
    cells = []
    id = 0
    for b,l in zip(boxes, labels):
        l = 'OHC' if l == 1 else 'IHC'
        c = Cell(id=id, parent=parent, score=1.0, bbox=b, type=l)
        id += 1
        cells.append(c)

    return cells