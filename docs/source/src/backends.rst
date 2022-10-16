hcat.backends
=====================
These backends accept a single channel volume from a confocal z-stack, and outputs a corresponding instance
segmentation mask of each hair cell in the volume.

.. warning::
   These backends apply internal checks on the volume to "guess" if any cells exist. These checks exist to expediate
   computationally expensive operations which is likely to **not** generate usable results. These checks may be
   overwritten by calling method: `no_reject()`.

Spatial Embedding
~~~~~~~~~~~~~~~~~
.. autoclass:: src.backends.spatial_embedding.SpatialEmbedding
   :members: forward, load, no_reject, reject

UNet + Watershed
~~~~~~~~~~~~~~~~
.. autoclass:: src.backends.unet_and_watershed.UNetWatershed
   :members: forward, load, no_reject, reject

Cellpose
~~~~~~~~
.. autoclass:: src.backends.cellpose.Cellpose
   :members: forward, no_reject, reject
