.. HCAT documentation master file, created by
   sphinx-quickstart on Thu Jul 29 13:05:30 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to HCAT's documentation!
================================

.. include:: src/introduction.md
   :parser: myst_parser.sphinx_

.. toctree::
   :caption: Instructions
   :maxdepth: 2

   src/installation.md
   src/usage_guide.md
   src/cli_instructions.md

.. toctree::
   :caption: Analysis Entrypoints:
   :maxdepth: 1

   detect

.. toctree::
   :caption: API Guide:
   :maxdepth: 1

   src/lib
   src/backend

