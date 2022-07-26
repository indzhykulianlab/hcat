import PySimpleGUI as sg
from hcat.lib.cell import Cell
from hcat.lib.cochlea import Cochlea
import skimage.io
import torchvision.utils
from PIL import Image
import io
from hcat.detect import _detect
import torch
from torch import Tensor
import numpy as np
from typing import Union, Optional, List, Dict, Tuple
import hcat.lib.utils
import torch.nn.functional as F
from torchvision.io import encode_png
from hcat.detect import _cell_nms
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, FigureCanvasAgg
import torchvision.transforms.functional as ttf

MAX_WIDTH, MAX_HEIGHT = 900, 900


# ------------------------------- END OF YOUR MATPLOTLIB CODE -------------------------------

# ------------------------------- Beginning of Matplotlib helper code -----------------------


class gui:
    def __init__(self):
        sg.theme('DarkGrey5')
        plt.ioff()
        # sg.set_options(font='Any')

        button_column = [
            [sg.FileBrowse(size=(15, 1), enable_events=True), ],
            [sg.Button('Load', size=(15, 1)), ],
            [sg.Button('Save', size=(15, 1))],
            [sg.HorizontalSeparator(p=(0, 20))],
            [sg.Text('Cell Diameter (In Pixels)')],
            [sg.Input(size=(10, 1), enable_events=True, default_text=30, key='Diameter'),
             sg.OK(size=(3, 1), key='change_size')],
            [sg.Text('Detection Threshold')],
            [sg.Slider(range=(0, 100), orientation='h', enable_events=True, default_value=80, key='Threshold')],
            [sg.Text('Overlap Threshold')],
            [sg.Slider(range=(0, 100), orientation='h', enable_events=True, default_value=30, key='NMS')],
            [sg.HorizontalSeparator(p=(0, 20))],
            [sg.Button('Run Analysis', size=(15, 1))],
            [sg.Check(text='Live Update: ', key='live_update', enable_events=True)]
        ]
        image_column = [
            [sg.Image(filename='', key='image', size=(900, 900))]
        ]

        adjustment_column = [
            [sg.Canvas(key='Histogram', border_width=2)],
            [sg.Text('Colors to include: '), sg.Check(k='Red', text='Red', default=True, enable_events=True),
             sg.Check(k='Green', text='Green', default=True, enable_events=True),
             sg.Check(k='Blue', text='Blue', default=True, enable_events=True)],
            [sg.Text('', pad=(0, 20))],

            [sg.Text(text='RED', text_color='#ff0000')],
            [sg.Text('Brightness'),
             sg.Slider(key='r_brightness', range=(0, 4), default_value=1, enable_events=True, orientation='h',
                       resolution=0.01, expand_x=True, size=(1, 10))],
            [sg.Text('Contrast  '),
             sg.Slider(key='r_contrast', range=(0, 2), default_value=1, enable_events=True, orientation='h',
                       resolution=0.01,
                       expand_x=True, size=(1, 10))],
            [sg.Text('', pad=(0, 20))],

            [sg.Text(text='GREEN', text_color='#00ff00')],
            [sg.Text('Brightness'),
             sg.Slider(key='g_brightness', range=(0, 4), default_value=1, enable_events=True, orientation='h',
                       resolution=0.01, expand_x=True, size=(1, 10))],
            [sg.Text('Contrast  '),
             sg.Slider(key='g_contrast', range=(0, 2), default_value=1, enable_events=True, orientation='h',
                       resolution=0.01,
                       expand_x=True, size=(1, 10))],

            [sg.Text('', pad=(0, 20))],
            [sg.Text(text='BLUE', text_color='#0000ff')],
            [sg.Text('Brightness'),
             sg.Slider(key='b_brightness', range=(0, 4), default_value=1, enable_events=True, orientation='h',
                       resolution=0.01, expand_x=True, size=(1, 10))],
            [sg.Text('Contrast  '),
             sg.Slider(key='b_contrast', range=(0, 2), default_value=1, enable_events=True, orientation='h',
                       resolution=0.01,
                       expand_x=True, size=(1, 10))],

            [sg.HorizontalSeparator(pad=(0, 30))],
            [sg.Text('OHC: None', key='OHC_count')],
            [sg.Text('IHC: None', key='IHC_count')],

        ]

        layout = [[sg.Column(button_column, vertical_alignment='Top'),
                   sg.VerticalSeparator(),
                   sg.Column(image_column),
                   sg.VerticalSeparator(),
                   sg.Column(adjustment_column, vertical_alignment='Top')
                   ]]

        self.window = sg.Window('HCAT', layout, finalize=True, return_keyboard_events=True)
        print(self.window)

        self.rgb_adjustments = ['r_brightness', 'r_contrast', 'g_brightness', 'g_contrast', 'b_brightness',
                                'b_contrast', ]

        for key in self.rgb_adjustments:
            self.window[key].bind('<ButtonRelease-1>', ' release')

        self.rgb_release = [x + ' release' for x in self.rgb_adjustments]

        # State
        self.__LOADED__ = False
        self.__DIAMETER__ = 30

        # Image Buffers
        self.raw_image = None  # raw image from file
        self.scaled_image = None  # to adjust cell diameter size
        self.scaled_and_adjusted_image = None

        self.display_image_scaled = None
        self.display_image = None
        self.scale_ratio = None

        self.fig_agg = None
        self.fig = None

        self.rgb = None
        self.contrast = None
        self.brightness = None
        self.model = None
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.cochlea = None

        self.all_boxes = None
        self.all_labels = None
        self.all_scores = None

        self.boxes = None
        self.labels = None
        self.scores = None


    def main_loop(self):

        while True:
            event, values = self.window.read(timeout=20)

            values[
                'Browse'] = '/home/chris/Dropbox (Partners HealthCare)/Gersten et al 2020 Fig 3 images/Fig7_max_projections adjusted/Carbo229_8kHz_63X_Maximum intensity projection.tif'

            self.rgb = [values['Red'], values['Green'], values['Blue']]
            self.contrast = [values['r_contrast'], values['g_contrast'], values['b_contrast']]
            self.brightness = [values['r_brightness'], values['g_brightness'], values['b_brightness']]

            if event == 'Exit' or event == sg.WIN_CLOSED:
                return

            if event == 'live_update' and values['live_update']:
                sg.popup_quick_message('Live model update enabled! May cause crashes on larger images...')

            # Load an image for the first time
            if event == 'Load' and values['Browse'] != '':
                self.load_image(values['Browse'])
                self.draw_image()
                self.render_hist()

                self.__LOADED__ = True

            # Update Histogram:
            if event in self.rgb_adjustments and self.__LOADED__:
                if values['live_update']:
                    self.run_detection_model()
                    self.threshold_and_nms(values['Threshold'], values['NMS'])

                self.draw_image()

            if event in ['Red', 'Green', 'Blue'] + [x + ' release' for x in self.rgb_adjustments] and self.__LOADED__:
                self.render_hist()

            if event == 'change_size' and self.__LOADED__:
                try:
                    self.__DIAMETER__ = float(values['Diameter'])
                except:
                    sg.popup_ok(
                        f'Cannot convert input: {values["Diameter"]} to a number. Defaulting to {self.__DIAMETER__}')

                if values['live_update']:
                    self.run_detection_model()
                    self.threshold_and_nms(values['Threshold'], values['NMS'])

                self.delete_fig_agg()
                self.load_image(values['Browse'])
                self.draw_image()
                self.render_hist()

            if event == 'Run Analysis' and not self.__LOADED__:
                sg.popup_quick_message('No File Loaded')

            if event == 'Run Analysis' and self.__LOADED__:
                self.run_detection_model()
                self.threshold_and_nms(values['Threshold'], values['NMS'])
                self.draw_image()

            if event in ['Threshold', 'NMS']:
                self.threshold_and_nms(values['Threshold'], values['NMS'])
                self.draw_image()

            if event == 'Save' and self.labels is not None:
                self.save(values['Threshold'], values['NMS'], values['Browse'])
                sg.popup_quick_message('Saved!')


    def run_detection_model(self):
        _image = self.scaled_image
        for i in range(3):
            if self.rgb[i]:
                _image[i] = ttf.adjust_brightness(_image[[i], ...], self.brightness[i])
                _image[i] = ttf.adjust_contrast(_image[[i], ...], self.contrast[i])
            else:
                _image[i] = _image[i, ...] * self.rgb[i]

        self.cochlea, self.model = _detect(image_base=_image[0:3, ...],
                                           cell_detection_threshold=0.0,
                                           nms_threshold=1.0,
                                           save_fig=False,
                                           save_png=False,
                                           model=self.model,
                                           return_model=True,
                                           )

        self.all_boxes: Tensor = torch.tensor([cell.boxes.cpu().tolist() for cell in self.cochlea.cells]) * self.ratio
        self.all_scores: Tensor = torch.tensor([cell.scores for cell in self.cochlea.cells])
        self.all_labels: List[str] = [cell.type for cell in self.cochlea.cells]

    @staticmethod
    def image_to_byte_array(image: Image) -> bytes:
        array = io.BytesIO()
        image.save(array, format='PNG')
        return array.getvalue()

    def load_image(self, f: str):

        self.clear_state()

        img: np.array = hcat.lib.utils.load(f, verbose=False)
        scale: int = hcat.lib.utils.get_dtype_offset(img.dtype, img.max())
        self.raw_image: Tensor = hcat.lib.utils.image_to_float(img, scale, verbose=False).to(self.device)
        self.scaled_image: Tensor = hcat.lib.utils.correct_pixel_size_image(self.raw_image, None,
                                                                            cell_diameter=float(self.__DIAMETER__),
                                                                            verbose=False).to(self.device)

        _, x, y = self.scaled_image.shape
        self.ratio = min(900 / x, 900 / y)
        self.display_image_scaled = F.interpolate(self.scaled_image.unsqueeze(0),
                                                  scale_factor=(self.ratio, self.ratio)).squeeze(0)
        print(f'Loaded image and scaled to shape {self.display_image_scaled.shape}')

    def clear_state(self):
        self.all_boxes = None
        self.all_labels = None
        self.all_scores = None

        self.boxes = None
        self.labels = None
        self.scores = None

    def threshold_and_nms(self, thr, nms_thr) -> Tensor:
        cell_key = {'OHC': '#56B4E9', 'IHC': '#E69F00'}

        ind = self.all_scores > (thr / 100)

        _labels = torch.tensor([1 if l == 'OHC' else 2 for l, i in zip(self.all_labels, ind) if i > 0])
        _boxes = self.all_boxes[ind, :]
        _scores = self.all_scores[ind]

        ind = torchvision.ops.nms(_boxes, _scores, nms_thr / 100)  # int indicies

        _boxes = _boxes[ind, :]
        _scores = _scores[ind]

        _labels = ['OHC' if _labels[i] == 1 else 'IHC' for i in ind]

        self.boxes = _boxes
        self.labels = _labels
        self.scores = _scores

        self.update_cell_counts()

    def get_color_histogram(self):
        color = ['r', 'g', 'b']
        fig = plt.figure(figsize=(3, 1), dpi=100)

        fig.patch.set_facecolor('#343434')
        fig.patch.set_alpha(1)

        fig.add_subplot(111)
        ax = plt.gca()

        for i in range(3):
            if self.rgb[i]:
                # hist = torch.histogram(self.display_image[i, ...].float())
                _, _, bars = ax.hist(self.display_image[i, ...].flatten().cpu().numpy(), color=color[i], bins=126,
                        density=True, alpha=0.7)

                max_height = max([b.get_height() for b in bars.patches])
                for b in bars.patches:
                    b.set_height(b.get_height() / max_height)


        plt.ylim([0, 1])
        plt.xlim([0, 255])

        ax.spines.right.set_visible(False)
        ax.spines.left.set_visible(False)
        ax.spines.top.set_visible(False)

        plt.tick_params(top=False, bottom=False, left=False, right=False,
                        labelleft=False, labelbottom=False)

        ax.patch.set_facecolor('#343434')
        plt.tight_layout()

        self.fig = plt.gcf()


    def delete_fig_agg(self):
        if self.fig_agg:
            self.fig_agg.get_tk_widget().forget()
            plt.close('all')
        self.window.refresh()

    def draw_figure(self):
        figure_canvas_agg = FigureCanvasTkAgg(self.fig, self.window['Histogram'].TKCanvas)
        figure_canvas_agg.draw()
        figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
        self.fig_agg = figure_canvas_agg

    def draw_image(self):
        cell_key = {'OHC': '#56B4E9', 'IHC': '#E69F00'}

        _image = self.display_image_scaled.clone()

        for i in range(3):
            if self.rgb[i]:
                _image[i] = ttf.adjust_brightness(_image[[i], ...], self.brightness[i])
                _image[i] = ttf.adjust_contrast(_image[[i], ...], self.contrast[i])

            _image[i] = _image[i, ...] * self.rgb[i]

        _image: Tensor = hcat.lib.utils.make_rgb(_image)
        _image: Tensor = _image.mul(255).round().to(torch.uint8).cpu()

        if self.labels:
            color = [cell_key[l] for l in self.labels]
            _image: Tensor = torchvision.utils.draw_bounding_boxes(image=_image,
                                                                boxes=self.boxes,
                                                                colors=color)

        img = encode_png(_image, 0).numpy().tobytes()
        _, x, y = _image.shape

        self.window['image'].update(data=img, size=(y, x))
        self.display_image = _image

    def render_hist(self):
        self.delete_fig_agg()
        self.get_color_histogram()
        self.draw_figure()

    def save(self, thr, nms, filename):
        _cells = [c for c in self.cochlea.cells if c.scores > thr]
        _cells = self.cell_nms(_cells, nms)
        self.cochlea.cells = _cells

        self.cochlea.write_csv(filename=filename)
        self.cochlea.make_detect_fig(self.scaled_image, filename=filename)

    def update_cell_counts(self):
        if self.labels:
            ihc = sum([l == 'IHC' for l in self.labels])
            ohc = sum([l == 'OHC' for l in self.labels])
        else:
            ihc, ohc = 'None', 'None'

        self.window['IHC_count'].update(f'IHC: {ihc}')
        self.window['OHC_count'].update(f'OHC: {ohc}')


    @staticmethod
    def cell_nms(cells: List[Cell], nms_threshold: float) -> List[Cell]:
        """
        Perforns non maximum supression on the resulting cell predictions

        :param cells: Iterable of cells
        :param nms_threshold: cell iou threshold
        :return: Iterable of cells
        """
        # nms to get rid of cells
        boxes = torch.zeros((len(cells), 4))
        scores = torch.zeros(len(cells))
        for i, c in enumerate(cells):
            boxes[i, :] = c.boxes
            scores[i] = c.scores

        ind = torchvision.ops.nms(boxes, scores, nms_threshold)

        # need to pop off list elements from an int64 tensor
        ind_bool = torch.zeros(len(cells))
        ind_bool[ind] = 1
        for i, val in enumerate(ind_bool):
            if val == 0:
                cells[i] = None

        return [c for c in cells if c]


if __name__ == '__main__':
    gui().main_loop()
