hcat.lib
========

cell
~~~~
The "cell" object is the base object for a detected cell. It contains all pertinent, cell specific,
information, such as classification, location, volume, or fluorescent intensity.

.. autoclass:: src.lib.cell.Cell
   :members:

cochlea
~~~~~~~
.. automodule:: src.lib.cochlea
   :members:

functional
~~~~~~~~~~
.. autoclass::src.lib.functional.EmbeddingToProbability
   :members:

.. autoclass::src.lib.functional.VectorToEmbedding
   :members:

.. autoclass::src.lib.functional.EstimateCentroids
   :members:

.. autoclass::src.lib.functional.nms
   :members:

.. autoclass:: src.lib.functional.GenerateSeedMap
   :members:

.. autoclass:: src.lib.functional.InstanceMaskFromProb
   :members:

.. autoclass:: src.lib.functional.PredictSemanticMask
   :members:

.. autoclass:: src.lib.functional.IntensityCellReject
   :members:

.. automodule:: src.lib.functional
    :members: get_cochlear_length, learnable_centroid, merge_regions

utils
~~~~~
.. automodule:: src.lib.utils
   :members: