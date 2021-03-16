.. _sec_cli:

======================
Command line interface
======================

``arginfer`` provides a Command Line Interface to access the
:ref:`Python API <sec_python_api>`.


.. code-block:: bash

    $ arginfer

or

.. code-block:: bash

    $ python3 -m arginfer

The second command is useful when multiple versions of Python are
installed or if the :command:`arginfer` executable is not installed on your path.

++++++++++++++++
Argument details
++++++++++++++++

.. argparse::
    :module: arginfer.cli
    :func: arginfer_cli_parser
    :prog: arginfer
    :nodefault:
