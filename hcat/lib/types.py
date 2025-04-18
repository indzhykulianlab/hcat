from typing import *
from torch import Tensor


class FilterParams(TypedDict):
    """parameters for a iir filter"""

    f0: float
    f1: float
    order: int


class Point(TypedDict):
    """ X Y points for a peak or notch """
    x: float
    y: float


WaveformPeaks = Dict[int, Point]  # WaveNumber: Point
WaveformNotches = Dict[int, Point]
WaveformAmplitudes = Dict[int, float]

ABRPeaks = Dict[int, WaveformPeaks]  # Level: Peaks
ABRNotches = Dict[int, WaveformNotches]
ABRAmplitudes = Dict[int, WaveformAmplitudes] # Level: WaveformAmplitudes

class ExperimentDict(TypedDict):
    waveforms: Tensor
    levels: List[int]
    threshold: Union[str, int]
    frequency: float
    filter_params: FilterParams
    peaks: ABRPeaks
    notches: ABRNotches


Store = Dict[str, ExperimentDict]


class Fields(TypedDict):
    attribute: str
    regex: str
    value: str


class FileData(TypedDict):
    filename: str
    fields: List[Fields]
