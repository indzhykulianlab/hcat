from hcat.detect import _detect
from hcat.detect_gui import gui
import os.path

import glob

import click
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

"""
This is the main entry point to all analysis scripts. Each analysis script is found in hcat.
"""


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    if ctx.invoked_subcommand is None:
        """ Default subcommand is running full analysis """
        gui().main_loop()

@cli.command()
@click.argument('f', default=None)
@click.option('--curve_path', default=None, help='CSV path of manually annotated cochlear curvature')
@click.option('--cell_detection_threshold', default=0.85, help='Threshold (between 0 and 1) of cell detection.')
@click.option('--nms_threshold', default=0.1, help='Allowable overlap between prediction boxes.')
@click.option('--dtype', default=None, help='dtype of input image')
@click.option('--save_xml', is_flag=True, help='Threshold (between 0 and 1) of cell detection.')
@click.option('--save_png', is_flag=True, help='Save a png for annoation purposes.')
@click.option('--save_fig', is_flag=True, help='Threshold (between 0 and 1) of cell detection.')
@click.option('--normalize', is_flag=True, help='Threshold (between 0 and 1) of cell detection.')
@click.option('--pixel_size', default=None, help='Pixel size in nm')
@click.option('--cell_diameter', default=None, help='Cell diameter in pixels')
@click.option('--predict_curvature', is_flag=True, help='Cell diameter in pixels')
@click.option('--silent', default=False, help="Suppresses most of HCAT's logging ")
def detect(f: str, curve_path, cell_detection_threshold, nms_threshold, save_xml, save_png,
           save_fig, normalize, pixel_size, dtype, cell_diameter, predict_curvature, silent):

    cell_diameter = float(cell_diameter) if cell_diameter is not None else None
    # Evaluate a single image

    verbose = not silent

    files = glob.glob(f)
    for filename in files:
        curve_path = filename[:-4:] + '_path.csv'
        print('CURVE PATH: ', curve_path)

        if not os.path.exists(filename[:-4:] + '_path.csv'):
            raise ValueError(curve_path)

        try:
            _detect(f=filename,
                    curve_path=curve_path,
                    cell_detection_threshold=cell_detection_threshold,
                    nms_threshold=nms_threshold,
                    save_png=save_png,
                    save_xml=save_xml,
                    save_fig=save_fig,
                    normalize=normalize,
                    pixel_size=pixel_size,
                    cell_diameter=cell_diameter,
                    dtype=dtype,
                    predict_curve=predict_curvature,
                    verbose=verbose)

        except Exception as e:
            print(f'Critical Error! Aborting - {filename}')
            raise e


if __name__ == '__main__':
    cli()
