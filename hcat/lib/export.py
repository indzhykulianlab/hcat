import os.path
import xml.etree.ElementTree as ET

from hcat.state import Piece, Cell


def piece_to_xml(piece: Piece) -> None:
    """
    Create an XML file of each cell detection from a cochlea object for use in the labelimg software.

    :param cochlea: Cochlea object from hcat.lib.cochlea.Cochlea
    :param filename: full file path by which to save the xml file

    :return: None
    """
    # Xml header and root
    filename = piece.filename

    folder = os.path.split(filename)[0]
    folder = os.path.split(folder)[-1] if len(folder) != 0 else os.path.split(os.getcwd())[1]

    _, height, width = piece.image.shape if piece.image is not None else (-1, -1, -1)
    depth = 1

    root = ET.Element('annotation')
    ET.SubElement(root, 'folder').text = folder
    ET.SubElement(root, 'filename').text = filename
    ET.SubElement(root, 'path').text = filename
    source = ET.SubElement(root, 'source')
    ET.SubElement(source, 'database').text = 'unknown'
    size = ET.SubElement(root, 'size')
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(height)
    ET.SubElement(size, 'depth').text = str(depth)
    ET.SubElement(root, 'segmented').text = '0'

    for c in piece.children.values():
        if not isinstance(c, Cell):
            pass

        x0, y0, x1, y1 = c.bbox
        #  xml write xmin, xmax, ymin, ymax
        object = ET.SubElement(root, 'object')
        ET.SubElement(object, 'name').text = c.type
        ET.SubElement(object, 'pose').text = 'Unspecified'
        ET.SubElement(object, 'truncated').text = '0'
        ET.SubElement(object, 'difficult').text = '0'
        bndbox = ET.SubElement(object, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(int(x0))
        ET.SubElement(bndbox, 'ymin').text = str(int(y0))
        ET.SubElement(bndbox, 'xmax').text = str(int(x1))
        ET.SubElement(bndbox, 'ymax').text = str(int(y1))

    tree = ET.ElementTree(root)
    filename = os.path.splitext(filename)[0]
    tree.write(filename + '.xml')