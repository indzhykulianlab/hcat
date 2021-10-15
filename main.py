from hcat.detect import _detect
from hcat.segment import _segment

import os.path
import glob

import click
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

"""
This is the main entry point to all analysis scripts. Each analysis script is found in src.
"""

@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    if ctx.invoked_subcommand is None:
        """ Default subcommand is running full analysis """
        segment()

@cli.command()
@click.argument('f', default=None)
@click.option('--channel', default=2, help='Channel index to segment')
@click.option('--intensity_reject_threshold', default=0.05, help='Cell cytosol intensity rejection threshold')
@click.option('--dtype', default=None, help='dtype of input image')
@click.option('--unet', is_flag=True, help='Run with Unet+Watershed Backbone')
@click.option('--cellpose', is_flag=True, help='Run with Cellpose Backbone')
@click.option('--figure', is_flag=True, help='Save preliminary analysis figure')
@click.option('--no_post', is_flag=True, help='Do not postprocess')
def segment(f: str, channel: int, intensity_reject_threshold: float,
            dtype: Optional[str],
            unet: bool, cellpose: bool, no_post: bool,
            figure: Optional[bool]) -> None:

    _segment(f=f, channel=channel, intensity_reject_threshold=intensity_reject_threshold,
             dtype=dtype, unet=unet, cellpose=cellpose, no_post=no_post, figure=figure)
    return None

@cli.command()
@click.argument('f', default=None)
@click.option('--curve_path', default=None, help='CSV path of manually annotated cochlear curvature')
@click.option('--cell_detection_threshold', default=0.85, help='Threshold (between 0 and 1) of cell detection.')
@click.option('--dtype', default=None, help='dtype of input image')
@click.option('--save_xml', is_flag=True, help='Threshold (between 0 and 1) of cell detection.')
@click.option('--save_fig', is_flag=True, help='Threshold (between 0 and 1) of cell detection.')
@click.option('--pixel_size', default=None, help='Pixel size in nm')
@click.option('--cell_diameter', default=None, help='Cell diameter in pixels')
@click.option('--batch', is_flag=True, help='Evaluate every image in folder')
def detect(f: str, curve_path, cell_detection_threshold, save_xml, save_fig, pixel_size, dtype, cell_diameter, batch):

    # Evaluate a single image
    if os.path.isfile(f):
        _detect(f=f, curve_path=curve_path, cell_detection_threshold=cell_detection_threshold,
                save_xml=save_xml, save_fig=save_fig, pixel_size=pixel_size, cell_diameter=cell_diameter, dtype=dtype)

    # Evaluate all images in a folder
    elif os.path.isdir(f) and batch:
        for f in glob.glob(f+'*.tif'):
            _detect(f=f, curve_path=curve_path, cell_detection_threshold=cell_detection_threshold,
                    save_xml=save_xml, save_fig=save_fig, pixel_size=pixel_size, cell_diameter=cell_diameter, dtype=dtype)
    else:
        print('\x1b[1;31;40m' + 'ERROR: Multi image cochlear analysis is currently not supported...' + '\x1b[0m')


if __name__ == '__main__':
    cli()
