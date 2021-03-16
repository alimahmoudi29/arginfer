.. _sec_tutorial:

========
Tutorial
========

*********************
Sampling ARGs
*********************

.. code-block:: python

    import msprime

    sample_ts = msprime.simulate(sample_size=10, Ne=10000,
                                            length=1e4,
                                            mutation_rate=1e-8,
                                            recombination_rate=1e-8,
                                            random_seed=2)
    print(sample_ts.num_trees,
          sample_ts.num_nodes)

The output of this code is:

.. code-block:: python

