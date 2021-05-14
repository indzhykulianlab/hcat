
# Have to do this first to avoid circular import. I am a bad programer.
class ShapeError(Exception):
    pass

from src.backends.spatial_embedding import SpatialEmbedding
from src.backends.unet_and_watershed import UNetWatershed

