import base64
import datetime
from typing import List, Dict

import numpy as np

from hcat.state import *


def cochlea_to_json(
    cochlea: Cochlea, style_hint: Dict | None = None, render_hint: Dict | None = None
) -> Dict:
    json_dict = {
        "pieces": [],
        "id": cochlea.id,
        "style_hint": style_hint,  # 'how we render the stuff annotations
        "render_hint": render_hint,  # 'how we render the image...
    }

    for piece_id, piece in cochlea.children.items():
        assert isinstance(piece, Piece)
        _piece = {
            "stateitem": "Piece",
            "user_set_name": piece.tree_item.text(0),
            "filename": piece.filename,
            "freq_path": piece.freq_path,
            "basal_freq": piece.basal_frequency,
            "apical_freq": piece.apical_frequency,
            "microscopy_type": piece.microscopy_type,
            "microscopy_zoom": piece.microscopy_zoom,
            "stain_label": piece.stain_label,
            "animal": piece.animal,
            "relative_order": piece.relative_order,
            "cell_thr": piece.cell_thr,
            "nms_thr": piece.nms_thr,
            "id": piece_id,
            "dtype": piece.dtype,
            "pixel_size_xy": piece.pixel_size_xy,
            "pixel_size_z": piece.pixel_size_z,
            "filepath": piece.filename,
            "eval_region": [(a.x(), a.y()) for a in piece.eval_region],
            "eval_region_boxes": piece.eval_region_boxes,
            "adjustment": piece.adjustments,
            "image": array_to_base64(piece.image),
            "image_shape": piece.image.shape,
            "children": [],
        }
        for cell_id, child in piece.children.items():
            if isinstance(child, Cell):
                _piece["children"].append(_cell_to_dict(child))
            elif isinstance(child, Synapse):
                raise ValueError

        for cell_id, child in piece._candidate_children.items():
            if isinstance(child, Cell):
                _piece["children"].append(_cell_to_dict(child))
            elif isinstance(child, Synapse):
                raise ValueError

        json_dict["pieces"].append(_piece)

    return json_dict


def _cell_to_dict(c: Cell):
    _dict = {
        "stateitem": "Cell",
        "type": c.type,
        "frequency": c.frequency,
        "bbox": c.bbox,
        "mask_verticies": c.mask_verticies,
        "id": c.id,
        "score": c.score,
        "ground_truth": c.groundTruth(),
        "line_to_path": c.line_to_closest_path,
        'creator': c.get_creator(),
        'image_adjustments_at_creation': c.get_image_adjustments_at_creation(),
        'has_been_edited': c.was_edited(),
    }
    return _dict


def array_to_base64(array) -> str:
    return base64.b64encode(array).decode("utf-8")



def export_to_csv(cochlea: Cochlea, filepath: str):
    header_0 = f'SAVE LOCATION: {filepath}\n'
    header_1 = f'SAVE TIME {datetime.datetime.now().strftime("(YYYY/MM/DD HH:MM:SS): %Y/%m/%d %H:%M:%S")}\n'
    print('EXPORTING!')
    keys = 'state_item,object_id,cell_type,frequency,percentage,distance(nm),score,bbox_x0,bbox_y0,bbox_x1,bbox_y1,center_x,center_y,' \
             'filename,piece_id,line_to_path,piece_relative_order,pixel_size_xy,piece_name,image_dtype,image_shape,image_sha256,' \
             'microscopy_type,microscopy_objective,animal,frequency_path,basal_frequency,apical_freqeucny,' \
             'creator,was_edited,is_ground_truth,channel_0_stain_label,channel_1_stain_label,channel_2_stain_label,channel_0_stain_color,' \
             'channel_1_stain_color,channel_2_stain_color,channel_0_brightness_at_creation,channel_0_contrast_at_creation,'\
             'channel_1_brightness_at_creation,channel_1_contrast_at_creation,channel_2_brightness_at_creation,channel_2_contrast_at_creation\n'
    with open(filepath, 'w') as csv:
        csv.write(header_0)
        csv.write(header_1)
        csv.write(keys)

        pieces: List[Piece] = [p for p in cochlea.children.values() if isinstance(p, Piece)]
        for piece in pieces:

            image_sha = str(piece.sha256_hash_of_image).replace(',', ' ')
            microscope_type = str(piece.microscopy_type).replace(',', ' ')
            microscope_zoom = str(piece.microscopy_zoom).replace(',', ' ')
            filename = str(piece.filename).replace(',', ' ')
            image_dtype = str(piece.image.dtype).replace(',', ' ')
            image_shape = str(piece.image.shape).replace(',', ' ')
            animal = str(piece.animal).replace(',', ' ')
            freq_path = str(piece.freq_path).replace(',', ' ')
            basal_freq = str(piece.basal_frequency).replace(',', ' ')
            apical_freq = str(piece.apical_frequency).replace(',', ' ')
            piece_name = str(piece.tree_item.text(0)).replace(',', ' ')
            px_size = str(piece.pixel_size_xy).replace(',', ' ')
            relative_order = str(piece.relative_order).replace(',', ' ')
            piece_id = str(piece.id).replace(',', ' ')
            channel_0_stain_label = str(piece.stain_label['red']).replace(',', ' ')
            channel_1_stain_label = str(piece.stain_label['green']).replace(',', ' ')
            channel_2_stain_label = str(piece.stain_label['blue']).replace(',', ' ')

            cells = [c for c in piece.children.values() if isinstance(c, Cell)]
            for c in cells:
                state_item = str('hair_cell').replace(',', ' ')
                identifier = str(c.id).replace(',', ' ')
                cell_type = str(c.type).replace(',', ' ')
                frequency = str(c.frequency).replace(',', ' ')
                percentage=str(c.percentage_total_length).replace(',', ' ')
                distance=str(c.distance_from_base_in_nm).replace(',', ' ')
                creator = str(c.get_creator()).replace(',', ' ')
                was_edited = str(c.was_edited()).replace(',', ' ')
                score = str(c.score).replace(',', ' ')
                x0,y0,x1,y1 = c.bbox
                cx, cy = c.get_cell_center()
                line_to_path = str(c.line_to_closest_path).replace(',', ' ')
                is_ground_truth = str(c.groundTruth()).replace(',', ' ')
                _adjust: List[Dict] = c.get_image_adjustments_at_creation()

                c0b = _adjust[0]['brightness'] if _adjust is not None else 'ERROR'
                c0c = _adjust[0]['contrast'] if _adjust is not None else 'ERROR'
                c1b = _adjust[1]['brightness'] if _adjust is not None else 'ERROR'
                c1c = _adjust[1]['contrast'] if _adjust is not None else 'ERROR'
                c2b = _adjust[2]['brightness'] if _adjust is not None else 'ERROR'
                c2c = _adjust[2]['contrast'] if _adjust is not None else 'ERROR'


                row = f'{state_item},{identifier},{cell_type},{frequency},{percentage},{distance},{score},{x0},{y0},{x1},{y1},{cx},{cy},' \
                      f'{filename},{piece_id},{line_to_path},{relative_order},{px_size},{piece_name},{image_dtype},{image_shape},{image_sha},' \
                      f'{microscope_type},{microscope_zoom},{animal},{freq_path},{basal_freq},{apical_freq},' \
                      f'{creator},{was_edited},{is_ground_truth},{channel_0_stain_label},{channel_1_stain_label},{channel_2_stain_label},{"(255 0 0)"},' \
                      f'{"(0 255 0)"},{"(0 0 255)"},{c0b},{c0c},' \
                      f'{c1b},{c1c},{c2b},{c2c}\n'

                csv.write(row)




if __name__ == "__main__":
    a = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.uint8)

    with open("test.txt", "w") as f:
        f.write(str(a.data))
