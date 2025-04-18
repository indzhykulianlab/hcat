import base64
import json
import os.path
from typing import Dict, Tuple

import numpy as np
from tqdm import tqdm

from hcat.state import Cell, Cochlea, Piece


def base64_to_array(string: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(string.encode('utf-8')), dtype=np.uint8)

def load_hcat_file(filepath: str) -> Tuple[Cochlea, Dict | None, Dict | None]:
    """ loads a json file from file and recreate the state """
    if not os.path.exists(filepath):
        return None

    with open(filepath, 'r') as f:
        json_dict = json.load(f)

    cochlea = Cochlea(id=json_dict['id'], parent=None)
    for pd in json_dict['pieces']:
        # print(f'{len(pd["image"])}')

        image: np.array = base64_to_array(pd['image'])
        image.resize(pd['image_shape'])

        filename = pd['filename']
        id = pd['id']

        piece = Piece(image, filename, id, cochlea)

        # set a bunch of incedentals
        piece.set_freq_path(pd['freq_path'])
        piece.set_microscopy_type(pd['microscopy_type'])
        piece.set_microscopy_zoom(pd['microscopy_zoom'])
        piece.set_animal(pd['animal'])
        piece.set_relative_order(pd['relative_order'])
        piece.set_thresholds(
            (pd['cell_thr'], pd['nms_thr'])
        )
        piece.set_stain_labels(pd['stain_label'])
        piece.set_xy_pixel_size(pd['pixel_size_xy'])
        piece.set_eval_region(pd['eval_region'])
        piece.set_eval_region_boxes(pd['eval_region_boxes'])
        piece.set_apical_freq(pd['apical_freq'])
        piece.set_basal_freq(pd['basal_freq'])
        piece.tree_item.setText(0, pd['user_set_name'])

        for cd in tqdm(pd['children'], total=len(pd['children'])):
            if cd['stateitem'] == 'Cell':

                cell = Cell(
                    id=cd['id'],
                    parent=piece,
                    score=cd['score'],
                    type=cd['type'],
                    frequency=cd['frequency'],
                    bbox=cd['bbox']
                )
                cell.setGroundTruth(cd['ground_truth'])
                cell.set_line_to_path(cd['line_to_path'])
                if 'image_adjustments_at_creation' in cd:
                    cell.set_image_adjustments_at_creation(cd['image_adjustments_at_creation'])
                if 'creator' in cd:
                    cell.set_creator(cd['creator'])
                if 'has_been_edited' in cd:
                    if cd['has_been_edited']:
                        cell.set_as_edited()

                piece.add_child(cell)

        cochlea.add_child(piece)
        cochlea.set_selected_children(piece.id)

        style_dict = json_dict['style_hint'] if 'style_hint' in json_dict else None
        render_hint = json_dict['render_hint'] if 'render_hint' in json_dict else None

    return cochlea, style_dict, render_hint

