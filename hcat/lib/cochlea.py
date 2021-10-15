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

import torchvision.transforms.functional

class Cochlea:
    def __init__(self,
                 mask: Tensor = None,
                 start_time: str = None,
                 analysis_time: str = None,
                 analysis_type: str = None, # Either 'detect' or 'segment'
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


        # gathered at runtime
        self.script_dir = os.getcwd() if script_dir is None else script_dir
        self.scripts = self._get_python_scripts(self.script_dir)

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
        match = re.search(' m\d ', filename)
        self._animal_id = match[0] if match is not None else None

    @property
    def litter(self):
        return self._litter

    @litter.setter
    def litter(self, filename):
        test_string = 'Oct 22 AAV2-PHP.B-CMV Olga L17 m1 G200 L0.25.lif'
        match = re.search(' L\d\d?\d?', filename)
        self._litter = match[0] if match is not None else None

    @property
    def gain(self):
        return self._gain

    @gain.setter
    def gain(self, filename):
        match = re.search('G\d\d?\d?', filename)
        self._gain = match[0] if match is not None else None

    @property
    def laser(self):
        return self.laser

    @laser.setter
    def laser(self, filename):
        match = re.search('L0.\d?\d?\d?', filename)
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

    def compress_mask(self, vebose: bool = False) -> None:
        """
        Compressed just the segmentation mask, which is usually the largest object in the class. This is usefull as it
        may be advantageous to load the class quickly without accessing the mask - for instance, getting metadata and
        nothing more.

        :param vebose: print compression statistics.
        :return: None
        """
        if isinstance(self.mask, Tensor):
            self.mask = self._compress_mask(self.mask, vebose)
        else:
            raise ValueError(f'Cannot compress mask of type {type(self.mask)}.')

    def decompress_mask(self) -> None:
        """
        Inverse operation to self.compress_mask. Will decompress self.mask from bytes to Tensor using the
        lz4 algorithm.

        :return: None
        """
        if isinstance(self.mask, bytes):
            self.mask = self._decompress_mask(self.mask)
        elif isinstance(self.mask, Tensor):
            raise ValueError(f'Cannot decompress mask of tpye {type(self.mask)}.')

    @staticmethod
    def _compress_mask(mask: Tensor, verbose: bool) -> bytes:
        """ compressed mask via lz4 algorithm """
        start = time.clock_gettime_ns(0)
        shape = mask.shape
        f = BytesIO()
        torch.save(mask, f)
        l = len(f.getbuffer())
        mask = compalg.compress(f.getbuffer())
        end = time.clock_gettime_ns(0)
        f.close()

        if verbose:
            print(f'Compressed mask with compression ratio of {l / len(mask)} in {(end - start) / 1e9} seconds')

        return mask

    @staticmethod
    def _decompress_mask(byte_mask: bytes) -> Tensor:
        return torch.load(BytesIO(compalg.decompress(byte_mask)))

    ####################################################################################################################
    #                                                    Analysis                                                      #
    ####################################################################################################################

    def render_mask(self, path: str,
                    random_color: Optional[bool] = False,
                    as_single_z_frames: Optional[Union[bool, str]] = False,
                    outline: Optional[bool] = False) -> None:
        """
        Renders tiff image output of predicted segmentation mask and saves to file.
        Can either render the volume mask with each pixel being a cell id, or a random color.

        :param path: path (including filename and extension) where to save the render
        :param random_color: if True, renders each cell with a unique color
        :param as_single_z_frames: NotImplemented
        :param outline: render only cell outlines instead of whole volume
        :return: None
        """

        was_compressed = False
        if isinstance(self.mask, bytes):
            self.decompress_mask()
            was_compressed = True
        c, x, y, z = self.mask.shape

        if outline:
            ind = torch.from_numpy(skimage.segmentation.find_boundaries(self.mask)).gt(0)
        else:
            ind = torch.ones(self.mask.shape).gt(0)

        # Rust speedup
        if random_color and not as_single_z_frames:
            colored_mask = torch.zeros((3, x, y, z), dtype=torch.int32)
            for i in range(3):
                colored_mask[i, ...][ind[0, ...]] = self.mask[ind][0, ...]

            print('Rendering Multicolored Tiff, may take a while...')
            t1 = time.clock_gettime_ns(0)
            colored_mask = self._render(colored_mask, self.num_cells)
            t2 = time.clock_gettime_ns(0)
            print(f'took {(t2 - t1) / 1e9} seconds')
            imsave(path, colored_mask.squeeze(0).numpy().transpose(3, 1, 2, 0, ))
        elif not random_color and not as_single_z_frames:
            imsave(path, self.mask.squeeze(0).int().numpy().astype(int8).transpose(2, 0, 1))
        # elif as_single_z_frames:
        #     os.mkdir(path)
        #     for i in range(z):

        if was_compressed:
            self.compress_mask()

    @staticmethod
    @torch.jit.script
    def _render(mask: Tensor, numcells: int) -> Tensor:
        for i in range(numcells):
            torch.random.manual_seed(i)
            color = torch.randint(low=0, high=255, size=(3,))
            index = mask == (i + 1)
            for i in range(3):
                mask[i, ...][index[0, ...]] = int(color[i])
        return mask

    @graceful_exit('\x1b[1;31;40m' + 'ERROR: csv generation failed.' + '\x1b[0m')
    def write_csv(self, filename: Optional[Union[bool, str]] = False) -> None:
        """
        Write results of cochlea object to a csv file for futher statistical analysis.

        .. warning:
           Will not raise an error upon failure, instead returns None and prints to standard out

        :param filename: filename to save csv as. If unset, uses image filename.
        :return: None
        """
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
        f.write(f'Treatment: {self.analysis_date}\n')
        f.write(label[:-1:] + '\n')  # index to remove final comma

        for cell in self.cells:
            f.write(f'{cell.id},{cell.frequency},{cell.percent_loc},')
            f.write(f'{cell.loc[1]},{cell.loc[2]},{cell.loc[3]},{cell.volume},{cell.summed},')

            for id in cell.channel_names:
                f.write(f'{cell.channel_stats[id]["mean"]},{cell.channel_stats[id]["median"]},{cell.channel_stats[id]["std"]},{cell.channel_stats[id]["var"]},')
                f.write(f'{cell.channel_stats[id]["min"]},{cell.channel_stats[id]["max"]},{cell.channel_stats[id]["%zero"]},{cell.channel_stats[id]["%saturated"]},')
            f.write('\n')
        f.close()

    # @graceful_exit('\x1b[1;31;40m' + 'ERROR: Figure Render failed.' + '\x1b[0m')
    def make_fig(self, filename: Optional[str] = None) -> None:
        """
        Make summary figure for quick interpretation of results.
        :param filename: filename to save figure as. If unset, uses image filename.
        :return: None
        """
        fig = plt.figure(figsize=(15, 8))

        ax = []
        x = [0.05, 0.35]
        y = [0.55, 0.075]
        w = 0.2
        h = 0.375
        for i in range(2):
            ax_ = []
            for j in range(2):
                ax_.append(fig.add_axes([x[i], y[j], w, h]))
            ax.append(ax_)

        hist_ax = []
        x = [0.05 + 0.2, 0.35 + 0.2]
        y = [0.55, 0.075]
        w = 0.05
        h = 0.375
        for i in range(2):
            ax_ = []
            for j in range(2):
                ax_.append(fig.add_axes([x[i], y[j], w, h]))
            hist_ax.append(ax_)

        im_ax = fig.add_axes([0.65, 0.075 * 2 + 0.25, 0.33, 0.55 - 0.075 / 2])
        cgram_ax = fig.add_axes([0.652, 0.075, 0.33, 0.25])

        num_cell = []
        # fig.set_size_inches(11,8)
        ind = [[0,0], [0,1], [1,0], [1,1]]


        for i, key in enumerate(self.cells[0].channel_names):
            signal = []
            perc = []
            x, y = ind[i]
            if i > 3:
                continue

            print(len(self.cells))
            for cell in self.cells:
                if cell.percent_loc is not None:
                    perc.append(cell.percent_loc)
                signal.append(cell.channel_stats[key]['mean'])


            if len(perc) != len(signal):
                perc = [i for i in range(len(signal))]

            perc = torch.tensor(perc) / max(perc)
            signal = torch.tensor(signal)
            channel = key
            signal = signal.div(signal.max()*2)

            mean_gfp = []
            try:
                for window in torch.linspace(0, 1, 20):
                    mean_gfp.append(signal[torch.logical_and(perc > window, perc < (window + 0.1))].mean())
            except IndexError:
                print(window, len(mean_gfp), signal.shape, perc.shape)

            stylestr = ['green', 'grey', 'blue', 'red']

            ax[x][y].plot(perc * 100, signal, 'ko', alpha=0.5)
            print(torch.linspace(0, 100, 20))
            ax[x][y].plot(torch.linspace(0, 100, 20), mean_gfp, c=stylestr[i], ls='-', lw=3)
            ax[x][y].set_xlabel('Cell Location (Cochlear % Apex -> Base)')
            ax[x][y].set_ylabel(f'Cell Mean {channel[i]} Intensity (AU)')
            ax[x][y].legend(['Cell'])
            ax[x][y].axhline(torch.mean(signal), c=stylestr[i], lw=1, ls='-')
            ax[x][y].axhline(torch.mean(signal) + torch.std(signal), c=stylestr[i], lw=1, ls='--', alpha=0.5)
            ax[x][y].axhline(torch.mean(signal) - torch.std(signal), c=stylestr[i], lw=1, ls='--', alpha=0.5)
            # ax[x][y].set_ylim([torch.mean(signal) - torch.std(signal)*3, torch.mean(signal) + torch.std(signal) * 3])

            hist_ax[x][y].hist(signal.numpy(), color=stylestr[i], bins=30, orientation='horizontal')
            hist_ax[x][y].spines['right'].set_visible(False)
            hist_ax[x][y].spines['top'].set_visible(False)
            # hist_ax[x][y].spines['bottom'].set_visible(False)
            hist_ax[x][y].spines['left'].set_visible(False)
            hist_ax[x][y].set_yticks([], minor=True)
            hist_ax[x][y].axis('off')


        # new = fig.add_axes([0, 0, 0.5, 0.5])
        # new.plot([0,1,2,3,4], [2,3,4,5,6])
        was_compressed = False
        if isinstance(self.mask, bytes):
            self.decompress_mask()
            was_compressed = True

        mask = self.mask.sum(-1).gt(0).squeeze(0)

        if was_compressed:
            self.compress_mask()

        im_ax.imshow(mask.numpy())

        if self.curvature is not None:
            im_ax.plot(self.curvature[0, :], self.curvature[1, :], lw=3)
            apex = self.curvature[:, self.cochlear_distance.argmax()]
            im_ax.plot(apex[0], apex[1], 'ro')
        im_ax.set_title('Max Projection Cell Masks')

        perc = []
        for cell in self.cells:
            if cell.percent_loc is not None:
                perc.append(cell.percent_loc)
        perc = torch.tensor(perc) / max(perc)

        fig.suptitle(os.path.split(self.filename)[-1])
        perc = perc.mul(100).float().round().div(100)
        cgram_ax.hist(perc.numpy(), bins=50, color='k')
        cgram_ax.set_title('Cochleogram')
        cgram_ax.set_xlabel('Cell Location (Cochlear % Apex -> Base)')
        cgram_ax.set_ylabel('Num Cells')

        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()

        return fig

    def make_detect_fig(self, image: Tensor, filename: Optional[str] = None):
        """
        image has to be torch.Tensor

        :param image:
        :param filename:
        :return:
        """
        print('Rendering Figure...', end='')
        image = image.mul(0.5).add(0.5).cpu() if image.min() < 0 else image.cpu()
        factor = 1.8

        while image.shape[0] < 3:
            zeromat = torch.zeros((1, image.shape[1], image.shape[2]))
            image = torch.cat((zeromat, image), dim=0)

        # image = torchvision.transforms.functional.adjust_brightness(image, factor)
        # image = torchvision.transforms.functional.adjust_contrast(image, factor*0.65)

        image[0,...] = image[0,...] * 0
        _, x, y = image.shape
        ratio = x / y
        fig = plt.figure(figsize=(y/200, x/200))
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        blue = '#2b75ff'

        ax.imshow(image[[2, 0, 1], ...].numpy().transpose((1, 2, 0)), cmap='bone')
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
        print('DONE')

    def make_cochleogram(self, filename: Optional[str] = None, type: Optional[str] = None):

        if self.curvature is None:
            print('\x1b[1;33;40mWARNING: ' +
                      'Predicted Cochlear Distance is below 4000um. Not sufficient information to generate cochleogram.'
                      + '\x1b[0m')
            return None

        def hist_coords(dist, nbin=50):
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
