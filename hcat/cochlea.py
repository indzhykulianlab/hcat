from __future__ import annotations

import torch
import time
import matplotlib.pyplot as plt
from typing import List, Union, Optional
from cell import Cell
from io import BytesIO

# import lzma as compalg
import lz4.frame as compalg

from xml.etree import ElementTree
import glob
import os.path
import re
from skimage.io import imsave
import skimage.segmentation


class Cochlea:
    """ Dataclass for containing results of a whole cochlea analysis. """
    def __init__(self,
                 mask: torch.Tensor = None,
                 start_time: str = None,
                 analysis_time: str = None,
                 filename: str = '',
                 script_dir: str = None,
                 leica_metadata: ElementTree = None,
                 im_shape: torch.Tensor = None,
                 cochlear_percent: torch.Tensor = None,
                 cochlear_length: torch.Tensor = None,
                 curvature: torch.Tensor = None,
                 apex: torch.Tensor = None,
                 cells: List[Cell] = None):

        self.analysis_time = analysis_time
        self.start_time = start_time
        self.analysis_date = time.asctime()
        self.filename = filename

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
        self.cochlear_percent = cochlear_percent
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

    def save(self, filename: str, compress: bool = False) -> None:
        """
        Meant to be invoked after class is full of data as a compressed bit object from torch.save

        :param filename: str | path and filename by which to save
        :param compress: bool | should object be compressed? Default is False. Mask is compressed automatically!
        :return:
        """
        if compress:
            self._save(self, filename)
        else:
            torch.save(self, filename)

    @classmethod
    def load(cls, filename: str) -> Cochlea:
        """
        Instantiate new class instance from a saved file.
        :param filename: str, path to file
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

        :param obj:
        :return:
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
        if isinstance(self.mask, torch.Tensor):
            self.mask = self._compress_mask(self.mask, vebose)
        else:
            raise ValueError(f'Cannot compress mask of type {type(self.mask)}.')

    def decompress_mask(self) -> None:
        """
        Inverse operation to self.compress_mask. Will decompress self.mask from bytes to torch.Tensor using the
        lz4 algorithm.

        :return:
        """
        if isinstance(self.mask, bytes):
            self.mask = self._decompress_mask(self.mask)
        elif isinstance(self.mask, torch.Tensor):
            raise ValueError(f'Cannot decompress mask of tpye {type(self.mask)}.')

    @staticmethod
    def _compress_mask(mask: torch.Tensor, verbose: bool) -> bytes:
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
    def _decompress_mask(byte_mask: bytes) -> torch.Tensor:
        return torch.load(BytesIO(compalg.decompress(byte_mask)))

    ####################################################################################################################
    #                                                    Analysis                                                      #
    ####################################################################################################################

    def render_mask(self, path: str, random_color: bool = False,
                    as_single_z_frames: Union[bool, str] = False, outline: bool = False) -> None:

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
                colored_mask[i, ...][ind[0,...]] = self.mask[ind][0, ...]

            print('Rendering Multicolored Tiff, may take a while...')
            t1 = time.clock_gettime_ns(0)
            colored_mask = self._render(colored_mask, self.num_cells)
            t2 = time.clock_gettime_ns(0)
            print(f'took {(t2 - t1) / 1e9} seconds')
            imsave(path, colored_mask.squeeze(0).numpy().transpose(3, 1, 2, 0, ))
        elif not random_color and not as_single_z_frames:
            imsave(path, self.mask.squeeze(0).div(self.mask.max()).numpy().transpose(2, 0, 1))
        # elif as_single_z_frames:
        #     os.mkdir(path)
        #     for i in range(z):

        if was_compressed:
            self.compress_mask()

    @staticmethod
    @torch.jit.script
    def _render(mask: torch.Tensor, numcells: int) -> torch.Tensor:
        for i in range(numcells):
            torch.random.manual_seed(i)
            color = torch.randint(low=0, high=255, size=(3,))
            index = mask == (i + 1)
            for i in range(3):
                mask[i, ...][index[0, ...]] = int(color[i])
        return mask

    def write_csv(self, filename: Union[bool, str] = False) -> None:
        """
        fixme: mean values fucked up, max and min reversed!!!!

        :param basedir:
        :return:
        """
        label = 'cellID,frequency,percent_loc,x_loc,y_loc,z_loc,volume,'
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
            f.write(f'{cell.loc[1]},{cell.loc[2]},{cell.loc[3]},{cell.volume},')

            # myo
            f.write(f'{cell.myo7a["mean"]},{cell.myo7a["median"]},{cell.myo7a["std"]},{cell.myo7a["var"]},')
            f.write(f'{cell.myo7a["min"]},{cell.myo7a["max"]},{cell.myo7a["%zero"]},{cell.myo7a["%saturated"]},')

            # dapi
            f.write(f'{cell.dapi["mean"]},{cell.dapi["median"]},{cell.dapi["std"]},{cell.dapi["var"]},')
            f.write(f'{cell.dapi["min"]},{cell.dapi["max"]},{cell.dapi["%zero"]},{cell.dapi["%saturated"]},')

            # actin
            f.write(f'{cell.actin["mean"]},{cell.actin["median"]},{cell.actin["std"]},{cell.actin["var"]},')
            f.write(f'{cell.actin["min"]},{cell.actin["max"]},{cell.actin["%zero"]},{cell.actin["%saturated"]},')

            # gfp
            f.write(f'{cell.gfp["mean"]},{cell.gfp["median"]},{cell.gfp["std"]},{cell.gfp["var"]},')
            f.write(f'{cell.gfp["min"]},{cell.gfp["max"]},{cell.gfp["%zero"]},{cell.gfp["%saturated"]}')
            f.write('\n')

    def make_fig(self, filename: Optional[str] = None) -> None:
        """
        Make summary figure for quick interpretation of results.

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

        # Cochleogram
        perc = []
        for cell in self.cells:
            perc.append(cell.percent_loc)
        num_cell = []
        perc = torch.tensor(perc)
        # fig.set_size_inches(11,8)
        i = 0
        for x in range(2):
            for y in range(2):
                signal = []
                perc = []

                for cell in self.cells:
                    if i == 0:
                        signal.append(cell.gfp['mean'])
                    elif i == 1:
                        signal.append(cell.myo7a['mean'])
                    elif i == 2:
                        signal.append(cell.dapi['mean'])
                    elif i == 3:
                        signal.append(cell.actin['mean'])

                    perc.append(cell.percent_loc)

                perc = torch.tensor(perc)
                signal = torch.tensor(signal)
                mean_gfp = []
                try:
                    for window in torch.linspace(0, 1, 20):
                        mean_gfp.append(signal[torch.logical_and(perc > window, perc < (window + 0.1))].mean())
                except IndexError:
                    print(window, len(mean_gfp), signal.shape, perc.shape)

                stylestr = ['g', 'y', 'b', 'r']
                channel = ['GFP', 'Myo7a', 'DAPI', 'Phalloidin']

                ax[x][y].plot(perc * 100, signal, 'ko')
                ax[x][y].plot(torch.linspace(0, 100, 20), mean_gfp, stylestr[i] + '-', lw=3)
                ax[x][y].set_xlabel('Cell Location (Cochlear % Apex -> Base)')
                ax[x][y].set_ylabel(f'Cell Mean {channel[i]} Intensity (AU)')
                ax[x][y].legend(['Cell'])
                ax[x][y].axhline(torch.mean(signal), c=stylestr[i], lw=1, ls='-')
                ax[x][y].axhline(torch.mean(signal) + torch.std(signal), c=stylestr[i], lw=1, ls='--', alpha=0.5)
                ax[x][y].axhline(torch.mean(signal) - torch.std(signal), c=stylestr[i], lw=1, ls='--', alpha=0.5)

                hist_ax[x][y].hist(signal.numpy(), color=stylestr[i], bins=30, orientation='horizontal')
                hist_ax[x][y].spines['right'].set_visible(False)
                hist_ax[x][y].spines['top'].set_visible(False)
                # hist_ax[x][y].spines['bottom'].set_visible(False)
                hist_ax[x][y].spines['left'].set_visible(False)
                hist_ax[x][y].set_yticks([], minor=True)
                hist_ax[x][y].axis('off')

                i += 1

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
        im_ax.plot(self.curvature[0, :], self.curvature[1, :], lw=3)
        apex = self.curvature[:, self.cochlear_percent.argmax()]
        im_ax.plot(apex[0], apex[1], 'ro')
        im_ax.set_title('Max Projection Cell Masks')

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


if __name__ == '__main__':
    a = torch.randint(0, 255, (1, 1000, 1000, 45))
    a[:, 2500::, :, :] = 0
    print('creating object')
    c = Cochlea(mask=a, cells=torch.arange(0, 255))
    c.render_mask('test.tiff')
