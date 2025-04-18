import numpy as np
from numba import jit


@jit(nopython=True, parallel=True)
def _adjust(img, brightness, contrast):
    """
    @jit(float64[:](float64[:], int64, float64[:]),nopython=True)
    def rnd1(x, decimals, out):
        return np.round_(x, decimals, out)


    :param img:
    :type img:
    :param brightness:
    :type brightness:
    :param contrast:
    :type contrast:

    :return:
    :rtype:
    """
    contrast = 10 ** (contrast / 200)
    # img = (img * contrast) + brightness
    out = np.zeros_like(img)
    np.round((img * contrast) + brightness, 0, out)  # cant have keywrods with round...
    return out.clip(0, 255).astype(np.uint8)