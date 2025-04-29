.. |logo|

..
# PyRFDPD: 
This package is used for digital predistortion algorithm.

Description
===========

*PyRFDPD* is a tailor-made python library for digital predistortion. 

Due to historical reasons, most algorithms and tools in the field of communication are implemented in MATLAB. However, this approach has a few drawbacks. 
#. MATLAB is not closed-source, and the MATLAB software is not free to use.
#. MATLAB toolkits often require official support, and lack of a community-driven ecosystem.
#. MATLAB is a standalone language and does not integrate well with other languages and tools.

On the other hand, Python, being an open-source language, offers a more flexible and accessible ecosystem for development. The industry may
sooner or latter shift to Python. That's what we have done, to accelerate this process.

|coverage| |tests_develop| |tests_master| |pypi| |license|

Installation
============

``$ python3 -m pip install pyrfdpd``
``$ python -m pip install -e ./pyrfdpd``

Kick-Start
==========

Cost Optimization
-----------------

.. code-block:: python


.. note::

For further details, see the |apidoc|_.

.. substitutions

.. .. |logo| image:: https://github.com/hahnec/torchimize/blob/develop/docs/torchimize_logo_font.svg
..     :target: https://hahnec.github.io/torchimize/
..     :width: 400 px
..     :scale: 100 %
..     :alt: torchimize
..
.. .. |coverage| image:: https://coveralls.io/repos/github/hahnec/torchimize/badge.svg?branch=master
..     :target: https://coveralls.io/github/hahnec/torchimize
..     :width: 98
..
.. .. |tests_develop| image:: https://img.shields.io/github/actions/workflow/status/hahnec/torchimize/gh_actions.yaml?branch=develop&style=square&label=develop
..     :target: https://github.com/hahnec/torchimize/actions/
..     :width: 105
..
.. .. |tests_master| image:: https://img.shields.io/github/actions/workflow/status/hahnec/torchimize/gh_actions.yaml?branch=master&style=square&label=master
..     :target: https://github.com/hahnec/torchimize/actions/
..     :width: 100
..
.. .. |license| image:: https://img.shields.io/badge/License-GPL%20v3.0-orange.svg?logoWidth=40
..     :target: https://www.gnu.org/licenses/gpl-3.0.en.html
..     :alt: License
..     :width: 150
..
.. .. |pypi| image:: https://img.shields.io/pypi/dm/torchimize?label=PyPI%20downloads
..     :target: https://pypi.org/project/torchimize/
..     :alt: PyPI Downloads
..     :width: 162
..
.. .. |apidoc| replace:: **API documentation**
.. .. _apidoc: https://hahnec.github.io/torchimize/build/html/apidoc.html

Citation
========

.. code-block:: BibTeX

    @misc{pyrfdpd,
        title={pyrfdpd},
        author={Microwave System Lab @ Southeast University},
        year={2024},
        publisher = {GitHub},
        journal = {GitHub repository},
        howpublished = {\url{https://github.com/SEU-MSLab/pyrfdpd}}
    }
