from __future__ import annotations

import torch
from torch import Tensor
import time
import matplotlib.pyplot as plt
from typing import List, Union, Optional, Iterable, Dict
from hcat.lib.cell import Cell
from hcat.lib.utils import graceful_exit
from io import BytesIO
from numpy import int8

import lz4.frame as compalg

from xml.etree import ElementTree
import glob
import os.path
import re
from skimage.io import imsave
import skimage.segmentation


class Cochlea:
    def __init__(self,
                 mask: Tensor = None,
                 start_time: str = None,
                 analysis_time: str = None,
                 analysis_type: str = None,  # Either 'detect' or 'segment'
                 filename: str = '',
                 path: str = '',
                 script_dir: str = None,
                 leica_metadata: ElementTree = None,
                 im_shape: Union[Tensor, tuple] = None,
                 cochlear_distance: Tensor = None,
                 cochlear_length: Tensor = None,
                 curvature: Tensor = None,
                 apex: Tensor = None,
                 cells: List[Cell] = None):
        """
        Dataclass of whole cochlear results.

        Useful for both storage, preliminary analysis, and processing of results.

        :cvar *float* gain: gain of detector at a particular channel
        :cvar *float* laser: laser intensity of a particular channel
        :cvar *str* litter: animal litter identification number
        :cvar *str* animal_id: animal identification number
        :cvar *str* analysis_time: total time of analysis
        :cvar *str* start_time: start time of analysis
        :cvar *str* filename: filename of image being analyzed
        :cvar *str* path: filepath of image being analyzed
        :cvar Iterable[float] voxel_size: voxel size (nm)
        :cvar Iterable[int] im_shape: shape of analyzed image
        :cvar List[Cell] cells: List of detected cell objects
        :cvar int num_cells: total number of detected objects
        :cvar Tensor mask: predicted instance segmentation mask
        :cvar Tensor curvature: 2D array of predicted cochlear curvature
        :cvar Tensor cochlear_length: length of cochlea, inferred from curvature
        :cvar Tensor cochlear_percent: percent length of cochlea, inferred from curvature
        :cvar Tensor apex: estimated apex of cochelea
        :cvar str script_dir: directory of analysis scripts used for this cochlea
        :cvar Dict[str,str] scripts: dictionary containing all analysis scripts

        :param mask: whole cochlea predicted segmentation mask
        :param start_time: start time of analysis
        :param analysis_time: total analysis time (seconds)
        :param filename: filename of analyzed image
        :param path: path to analyzed file
        :param script_dir: directory containing analysis scripts
        :param leica_metadata: parsed xml ElementTree from lif file
        :param im_shape: shape of analyzed image
        :param cochlear_percent: predicted array of cochlear percentage corresponding to curvature array
        :param cochlear_length: predicted cochlear length in mm
        :param curvature: predicted cochlear curvature
        :param apex: predicted apex location
        :param cells: list of each predicted cell object
        """

        self.analysis_time = analysis_time
        self.start_time = start_time
        self.analysis_date = time.asctime()
        self.analysis_type = analysis_type
        self.filename = filename
        self.path = path

        # Image params from filename (Semiprivate)
        self._gain = None
        self._laser = None
        self._promoter = None
        self._litter = None
        self._animal_id = None

        # Inferred from filename
        # Setters perform regex on filename and assign appropriate value
        # If no match is found, sets val at None
        self.gain: str = filename
        self.laser: str = filename
        self.promoter: str = filename  # Not Implemented!!!
        self.litter: str = filename
        self.animal_id: str = filename

        self.leica_metadata = leica_metadata

        # Infered from metadata (semiprivate)
        # NOT IMPLEMENTED!!!
        self._x_pix_size = None
        self._y_pix_size = None
        self._z_pix_size = None

        # Infered from metadata
        # NOT IMPLEMENTED!!!
        self.x_pix_size = None
        self.y_pix_size = None
        self.z_pix_size = None

        # Cochlear Analysis
        self.im_shape = im_shape
        self.cells: List[Cell] = cells if cells is not None else []
        self.num_cells = len(self.cells)
        self.mask = mask
        self.curvature = curvature
        self.cochlear_length = cochlear_length
        self.cochlear_distance = cochlear_distance
        self.apex = apex

        # # gathered at runtime
        # self.script_dir = os.getcwd() if script_dir is None else script_dir
        # self.scripts = self._get_python_scripts(self.script_dir)

    ####################################################################################################################
    #                                                Getters and Setters                                               #
    ####################################################################################################################

    @staticmethod
    def _get_python_scripts(script_dir):
        python_files = {}
        script_dir = os.path.join(script_dir, '')
        python_files_list = glob.glob(script_dir + '**/*.py', recursive=True)
        ipy_files_list = glob.glob(script_dir + '**/*.ipynb', recursive=True)

        for f in python_files_list:
            file = open(f, 'r')
            python_files[f] = file.read()
            file.close()

        for f in ipy_files_list:
            file = open(f, 'r')
            python_files[f] = file.read()
            file.close()

        return python_files

    @property
    def animal_id(self):
        return self._litter

    @animal_id.setter
    def animal_id(self, filename):
        match = re.search(' m\d ', filename) if filename else None
        self._animal_id = match[0] if match is not None else None

    @property
    def litter(self):
        return self._litter

    @litter.setter
    def litter(self, filename):
        test_string = 'Oct 22 AAV2-PHP.B-CMV Olga L17 m1 G200 L0.25.lif'
        match = re.search(' L\d\d?\d?', filename) if filename else None
        self._litter = match[0] if match is not None else None

    @property
    def gain(self):
        return self._gain

    @gain.setter
    def gain(self, filename):
        match = re.search('G\d\d?\d?', filename) if filename else None
        self._gain = match[0] if match is not None else None

    @property
    def laser(self):
        return self.laser

    @laser.setter
    def laser(self, filename):
        match = re.search('L0.\d?\d?\d?', filename) if filename else None
        self._laser = match[0] if match is not None else None

    ####################################################################################################################
    #                                                Compression and Saving                                            #
    ####################################################################################################################

    def save(self, filename: str, compress: Optional[bool] = False) -> None:
        """
        Meant to be invoked after class is full of data as a compressed bit object from torch.save

        :param filename: path and filename by which to save
        :param compress: should object be compressed? Default is False. Mask is compressed automatically!
        :return: None
        """
        if compress:
            self._save(self, filename)
        else:
            torch.save(self, filename)

    @classmethod
    def load(cls, filename: str) -> Cochlea:
        """
        Instantiate new class instance from a saved file.

        :param filename: path to file
        :return: Cochlea object
        """
        try:
            return cls._load(filename)
        except Exception:  # Throws _pickle.UnpicklingError if uncompressed
            return torch.load(filename)

    @staticmethod
    def _save(obj, filename: str):
        """
        Saves any object in a compressed file format

        :param obj: python object
        :return: None
        """
        f = BytesIO()
        torch.save(obj, f)
        file = open(filename, 'wb')
        out = compalg.compress(f.getbuffer())
        file.write(out)
        file.close()

    @staticmethod
    def _load(filename):
        file = compalg.open(filename, 'rb')
        return torch.load(file)

    # @graceful_exit('\x1b[1;31;40m' + 'ERROR: csv generation failed.' + '\x1b[0m')
    def write_csv(self, filename: Optional[Union[bool, str]] = None) -> None:
        """
        Write results of cochlea object to a csv file for futher statistical analysis.

        .. warning:
           Will not raise an error upon failure, instead returns None and prints to standard out

        :param filename: filename to save csv as. If unset, uses image filename.
        :return: None
        """
        if self.analysis_type == 'segment':
            label = 'cellID,frequency,percent_loc,x_loc,y_loc,z_loc,volume,summed,'
            for c in ['myo', 'dapi', 'actin', 'gfp']:
                label += f'{c}_mean,{c}_median,{c}_std,{c}_var,{c}_min,{c}_max,{c}_%zero,{c}_%saturated,'

            # print(filename)

            if filename is None and self.filename is not None:
                filename = os.path.splitext(self.filename)[0] + '.csv'  # Remove .lif and add .csv
            elif filename is None and self.filename is None:
                filename = 'analysis.csv'

            f = open(filename, 'w')
            f.write(f'Filename: {self.filename}\n')
            f.write(f'Analysis Date: {self.analysis_date}\n')
            f.write(label[:-1:] + '\n')  # index to remove final comma

            for cell in self.cells:
                f.write(f'{cell.id},{cell.frequency},{cell.percent_loc},')
                f.write(f'{cell.loc[1]},{cell.loc[2]},{cell.loc[3]},{cell.volume},{cell.summed},')

                for id in cell.channel_names:
                    f.write(
                        f'{cell.channel_stats[id]["mean"]},{cell.channel_stats[id]["median"]},{cell.channel_stats[id]["std"]},{cell.channel_stats[id]["var"]},')
                    f.write(
                        f'{cell.channel_stats[id]["min"]},{cell.channel_stats[id]["max"]},{cell.channel_stats[id]["%zero"]},{cell.channel_stats[id]["%saturated"]},')
                f.write('\n')
            f.close()
        elif self.analysis_type == 'detect':
            label = 'cellID,type,score,frequency,percent_loc,x_loc,y_loc'

            if filename is None and self.path is not None:
                filename = os.path.splitext(self.path)[0] + '.csv'  # Remove .lif and add .csv
            elif filename is None and self.path is None:
                filename = 'analysis.csv'

            filename = filename if filename.endswith('.csv') else os.path.splitext(filename)[0] + '.csv'

            f = open(filename, 'w')
            f.write(f'Filename: {self.filename}\n')
            f.write(f'Analysis Date: {self.analysis_date}\n')
            f.write(label[:-1:] + '\n')  # index to remove final comma

            for cell in self.cells:
                f.write(f'{cell.id},{cell.type},{cell.scores},{cell.frequency},{cell.percent_loc},')
                f.write(f'{cell.loc[1]},{cell.loc[2]}')
                f.write('\n')
            f.close()

    def make_detect_fig(self, image: Tensor, filename: Optional[str] = None):
        """
        Renders the summary figure of the HCAT detection analysis.

        :param image: torch.Tensor image which cells will be rendered on
        :param filename: filename by which to save the figure
        :return: None
        """
        image = image.cpu() if image.min() < 0 else image.cpu()

        while image.shape[0] < 3:
            zeromat = torch.zeros((1, image.shape[1], image.shape[2]))
            image = torch.cat((zeromat, image), dim=0)

        # image = torchvision.transforms.functional.adjust_brightness(image, factor)
        # image = torchvision.transforms.functional.adjust_contrast(image, factor*0.65)

        _, x, y = image.shape
        ratio = x / y
        fig = plt.figure(figsize=(y / 200, x / 200))
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        blue = '#2b75ff'

        ax.imshow(image.numpy().transpose((1, 2, 0)))
        #
        # Loop over all cells and plot them on an image
        if self.curvature is not None:
            ax.plot(self.curvature[0, :].numpy(), self.curvature[1, :].numpy(), '-', color=blue)
            for i, cell in enumerate(self.cells):
                _, x0, y0, _ = cell.loc
                cell.calculate_frequency(self.curvature, self.cochlear_distance)
                x1, y1 = self.curvature[:, cell._curve_ind]

                marker = 's' if cell.type == 'IHC' else 'o'
                # color = '#ff2b2b' if marker == 's' else '#36ff3c'  # 1B512D
                color = '#FFFF00' if marker == 's' else '#66ff00'  # 1B512D
                ax.plot([x0, x1], [y0, y1], '-', color=blue, linewidth=0.3, alpha=0.66)
                ax.plot([x0], [y0], marker, color=color, markersize=1)
                f1 = cell.frequency

                # Put diagnostic text on the image
                s = f'ID: {cell.id}\nFreq: {f1:.2f} kHz\n{self.cochlear_distance[cell._curve_ind]:.2f} μm'
                ax.text(x0, y0, s, fontsize=1.5, c='w')
        else:
            for i, cell in enumerate(self.cells):
                _, x0, y0, _ = cell.loc
                marker = 's' if cell.type == 'IHC' else 'o'
                # color = '#ff2b2b' if marker == 's' else '#36ff3c'  # 1B512D
                color = '#FFFF00' if marker == 's' else '#66ff00'  # 1B512D
                ax.plot([x0], [y0], marker, color=color, markersize=1)

                # Put diagnostic text on the image
                s = f'ID: {cell.id}'
                ax.text(x0, y0, s, fontsize=1.5, c='w')

            # # What is the cell freq from ground truth
        if filename is not None:
            fig.savefig(filename, dpi=400)
        else:
            fig.savefig(self.path[:-4:] + '.jpg', dpi=400)
        plt.close(fig)

    def make_cochleogram(self, filename: Optional[str] = None, type: Optional[str] = None, bin_size: Optional[int] = 2):
        """
        Generates a cochleogram from a detection analysis and saves it to a figure.
        Does nothing if the cochlear distance is less than 4000um.

        :param filename: Filename to save the cochleogram figure. Defaults to the base path of the cochlea object.
        :param type: Unused.
        :param bin_size: Cochleogram bin size in percentage total length [0 -> 100]
        :return: None
        """

        if self.curvature is None:
            print('\x1b[1;33;40mWARNING: ' +
                  'Predicted Cochlear Distance is below 4000um. Not sufficient information to generate cochleogram.'
                  + '\x1b[0m')
            return None

        def hist_coords(dist, nbin=100 / bin_size):
            dist = torch.tensor(dist)
            hist = torch.histc(dist.float(), nbin)

            b = torch.linspace(dist.min(), dist.max(), nbin)
            b = torch.linspace(dist.min(), b[-2].item(), nbin)

            bin_x = [dist.min(), b[0].item()]
            bin_y = [0, hist[0].item()]
            for i in range(1, len(b)):
                bin_x.append(b[i].item())
                bin_x.append(b[i].item())
                bin_y.append(hist[i - 1].item())
                bin_y.append(hist[i].item())
            bin_x.append(dist.max())
            bin_x.append(dist.max())
            bin_y.append(bin_y[-1])
            bin_y.append(0)
            return bin_x, bin_y

        nbin = 100
        plt.figure(figsize=(4, 2))
        plt.ylim([0, 50])
        # GT curve
        ###############
        dist = []
        for cell in self.cells:
            cell.calculate_frequency(self.curvature, self.cochlear_distance)
            if type is not None and cell.type == type:
                dist.append(cell.distance)
            elif type is None:
                dist.append(cell.distance)

        x, y = hist_coords(dist, nbin)
        plt.plot(x, y, '-')
        plt.xlabel(r'Cochlear Distance $Base\ →\ Apex\ (\mu m)$')
        plt.ylabel('Hair Cells')
        # plt.legend(['Ground Truth', 'Predicted'], frameon=False)
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename, dpi=450)
        else:
            plt.savefig(self.path[:-4:] + '_cochleogram.pdf', dpi=450)

        fig = plt.gcf()
        plt.close(fig)

        return fig


if __name__ == '__main__':
    a = torch.randint(0, 255, (1, 1000, 1000, 45))
    a[:, 2500::, :, :] = 0
    print('creating object')
    c = Cochlea(mask=a, cells=torch.arange(0, 255))
    c.render_mask('test.tiff')
