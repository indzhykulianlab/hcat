
# Have to do this first to avoid circular import. I am a bad programer.
class ShapeError(Exception):
    pass

from hcat.backends.spatial_embedding import SpatialEmbedding
from hcat.backends.unet_and_watershed import UNetWatershed

