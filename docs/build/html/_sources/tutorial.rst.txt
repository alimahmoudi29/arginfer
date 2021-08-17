.. _sec_tutorial:

========
Tutorial
========

*********************
Sampling ARGs
*********************
As a simple example, we will first simulate sample
data with  `msprime <https://tskit.dev/msprime/docs/stable/>`_.
We will then run `arginfer` on the simulated dataset.

The following code simulates a tree sequence and the sequences for a sample size of `10` and sequence
length of `1e5`.

.. code-block:: python

    import msprime
    import os
    ts_full = msprime.simulate(sample_size=10, Ne=5000,
                                            length=1e5,
                                            mutation_rate=1e-8,
                                            recombination_rate=0.5e-8,
                                            record_full_arg= True,
                                            random_seed=2)
    os.makedirs(os.getcwd()+"/out")
    ts_full.dump(os.getcwd()+"/out/"+"ts_full.args")


The output of this code is a ``tree sequence`` stored in "out/" directory under the name of `ts_full.args`.

Next,  the following command can
be used to run 200 MCMC iterations with burn-in 5 and retaining every 10 samples (thinning intervals = 10).
Also ``sample_size = n = 10`` is the number of sequences each ``seq_length = L = 1e5`` in length evolving in
a population of effective size ``Ne = 5000``, with
mutation rate ``1e-8`` mutations/generation/site and recombination rate ``0.5e-8``
recombinations/generation/site.

.. code-block:: python

    import arginfer
    arginfer.infer_sim(
        ts_full = "out/ts_full.args",     # path to simulated ts
        sample_size =10,            # sample size
        iteration= 200,              # number of mcmc iterations
        thin= 10,                    # thinning interval, retaining everry kth sample
        burn=5,                     # burn-in period to discard
        Ne =5000,                   # effective population size
        seq_length= 1e5,            # sequence length in bases
        mutation_rate=1e-8,         # mutation rate per site per generation
        recombination_rate=0.5e-8,    # recombination rate per site per generation
        outpath = os.getcwd()+"/output",   # output path
        plot = True)                    # plot traces

or equivalently in terminal:

.. code-block:: RST

    arginfer infer --tsfull "out/ts_full.args" \
        -I 200 --thin 10 -b 5 \
        -n 10 -L 1e5 --Ne 5000 \
        -r 0.5e-8 -mu 1e-8 \
        -O output \
        --plot

The output of the above command is as follows:

* ``summary.h5``: A summary of some ARG properties recorded in a ``pandas dataframe`` with columns:

.. code-block:: python

    pd.DataFrame(columns=('likelihood', 'prior', "posterior",
                                             'ancestral recomb', 'non ancestral recomb',
                                                'branch length'))

* ``.arg`` file: The sampled ARGs, which are pickled ``ATS`` objects.

    *  See here for more information on how manipulate these files (TODO).

*  | ``arginfer*.pdf``: if ``plot=True``, this `pdf` file will be generated which contains trace plots for
   |  the log(posterior), ARG total branch length, number of ancestral recombinations,
   |  and number of non-ancestral recombinations.


*********************************
Working with ``arginfer`` outputs
*********************************

