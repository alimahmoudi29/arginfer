#!/usr/bin/env python3
"""
Various functions to convert a ts file to ARGweaver input format,
and from .arg files to tree seq input.
When run as a script, takes an msprime simulation in .trees format, saves to
ARGweaver input format (haplotype sequences), runs ARGweaver inference on it to
make .smc files.

python3 AW_in.py --trees_file ts.trees \
    -d  /Users/amahmoudi/Dropbox/PhDMelbourne/Paper/ARGweaver/argweaver-master/bin \
    -x arg-sample --iterations 3 --sample_step 2 \
    --full_prefix /Users/amahmoudi/Ali/phd/github_projects/mcmc/test/alaki
    --trees_file ts.trees -Ne 5000 -mu 1e-8 -rho 1e-8
"""
import sys
import subprocess
import logging
import math
import re
import gzip
import csv
import os.path
import pandas as pd
import numpy as np
import time
import msprime
import shutil
f_dir = os.path.dirname(os.getcwd())+"/ARGinfer"
sys.path.append(f_dir)
# print(sys.path)
import treeSequence

import plot

def ts_to_ARGweaver_in(ts, ARGweaver_filehandle):
    """
    Takes a TreeSequence, and outputs a file in .sites format, suitable for input
    into ARGweaver (see http://mdrasmus.github.io/argweaver/doc/#sec-file-sites)
    The documentation (http://mdrasmus.github.io/argweaver/doc/#sec-prog-arg-sample)
    states that the only mutation model is Jukes-Cantor (i.e. equal mutation between
    all bases). Assuming adjacent sites are treated independently, we convert variant
    format (0,1) to sequence format (A, T, G, C) by simply converting 0->A and 1->T
    Msprime simulations assume infinite sites by allowing mutations to occur at
    floating-point positions along a sequence. ARGweaver has discrete sites instead.
    This routine implements a basic discretising function, which simply rounds upwards
    to the nearest int, ANDing the results if 2 or more variants end up at the same
    integer position.
    Note that ARGweaver uses position coordinates (1,N) - i.e. [0,N).
    That compares to tree sequences which use (0..N-1) - i.e. (0,N].
    """
    simple_ts = ts.simplify()
    print("\t".join(["NAMES"]+[str(x) for x in range(simple_ts.get_sample_size())]), file=ARGweaver_filehandle)
    print("\t".join(["REGION", "chr", "1", str(int(simple_ts.get_sequence_length()))]), file=ARGweaver_filehandle)
    genotypes = None
    position = 0
    for v in simple_ts.variants():
        if int(math.ceil(v.position)) != position:
            #this is a new position. Print the genotype at the old position, and then reset everything
            if position:
                print(position, "".join(np.where(genotypes==0,"A","T")), sep="\t", file=ARGweaver_filehandle)
            genotypes = v.genotypes
            position = int(math.ceil(v.position))
        else:
            genotypes = np.logical_and(genotypes, v.genotypes)
    if position:
        print(position, "".join(np.where(genotypes==0,"A","T")), sep="\t", file=ARGweaver_filehandle)

    ARGweaver_filehandle.flush()
    ARGweaver_filehandle.seek(0)

def tsfile_to_ARGweaver_in(trees, ARGweaver_filehandle, out_path):
    """
    take a .trees file, and convert it into an input file suitable for ARGweaver
    Returns the simulation parameters (Ne, mu, r) used to create the .trees file
    """
    logging.info("== Saving to ARGweaver input format ==")
    try:
        ts = msprime.load(trees.name) #trees is a fh
    except AttributeError:
        ts = msprime.load(trees)
    ts_to_ARGweaver_in(ts, ARGweaver_filehandle)
    # true values
    tsarg = treeSequence.TreeSeq(ts)
    tsarg.ts_to_argnode()
    data = treeSequence.get_arg_genotype(ts)
    arg = tsarg.arg
    print("true number of rec", len(arg.rec)/2)
    log_lk = arg.log_likelihood(1e-8, data)
    log_prior = arg.log_prior(ts.sample_size,
                              ts.sequence_length, 1e-8, 5000, False)
    np.save(out_path+"/true_values.npy", [log_lk, log_prior,
                                              log_lk + log_prior,
                                             arg.branch_length,
                                              arg.num_ancestral_recomb,
                                             arg.num_nonancestral_recomb,
                                              1e-8,# mu
                                              1e-8, 5000])# r, Ne
    #---
    #here we should extract the /provenance information from the .trees file and return
    # {'Ne':XXX, 'mutation_rate':XXX, 'recombination_rate':XXX}
    #but this information is currently not encoded in the .trees file (listed as TODO)

    return {'Ne':None, 'mutation_rate':None, 'recombination_rate':None}

ts = msprime.simulate(sample_size= 5, length=1e4, Ne=5000,
                      recombination_rate=1e-8, mutation_rate=1e-8, random_seed=1)

def main(args):
    import os
    import subprocess
    if not os.path.exists(args.full_prefix):
            os.makedirs(args.full_prefix)
    else:
        shutil.rmtree(args.full_prefix)
        os.mkdir(args.full_prefix)
    output= args.full_prefix
    args.full_prefix = args.full_prefix +"/"+"out"
    pre_time=time.time()

    with open(args.full_prefix+".sites", "w+") as aw_in:
        tsfile_to_ARGweaver_in(args.trees_file, aw_in,output)
        cmd = [os.path.join(args.ARGweaver_executable_dir,
                            args.ARGweaver_sample_executable),
            '--sites', aw_in.name,
            '--popsize', str(args.effective_population_size),
            '--recombrate', str(args.recombination_rate),
            '--mutrate', str(args.mutation_rate),
            '--overwrite',
            '--randseed', str(int(args.random_seed)),
            '--infsites',
            '--ntimes', str(40),
            '--compress-seq', str(1),
            '--iters', str(args.iterations),
            '--sample-step', str(args.sample_step),
            '--output', args.full_prefix]#the prefix for the output
        assert os.stat(aw_in.name).st_size > 0, "Initial .sites file is empty"
        logging.debug("running '{}'".format(" ".join(cmd)))
        subprocess.call(cmd)
        smc = args.full_prefix + "." + str(args.iterations) + ".smc.gz"
        # assert os.path.isfile(smc),  "No output file names {}".format(smc)
    end_time = time.time()
    np.save(args.full_prefix+"_time",[end_time-pre_time])
    if args.plot:
        p = plot.Trace(output, argweaver= True)
        p.argweaver_trace()

def aw_infer():
    import argparse
    import filecmp
    import os
    parser = argparse.ArgumentParser(prog="AWinfer",
            description='Check ARGweaver imports by running an msprime simulation to create an'
            'ARGweaver import file, inferring some args from it in smc format, converting the .smc format to .argformat,'
            'reading the .arg into msprime, and comparing the nexus output trees with the trees in the .smc file. '
                                                 'This testing process requires the dendropy library')
    parser.add_argument('--trees_file', type=argparse.FileType('r', encoding='UTF-8'), default=None,
                                            help='an msprime .trees file. If none, simulate one with defaults')
    parser.add_argument('--ARGweaver_executable_dir', '-d',
        default= "/Users/amahmoudi/Dropbox/PhDMelbourne/Paper/ARGweaver/argweaver-master/bin/",
                        #os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','argweaver/bin/')
        help='the path to the directory containing the ARGweaver executables')
    parser.add_argument('--ARGweaver_sample_executable', '-x', default="arg-sample",
                        help='the name of the ARGweaver executable')
    parser.add_argument('--ARGweaver_smc2arg_executable', '-s', default="smc2arg",
                        help='the name of the ARGweaver executable')
    parser.add_argument('--sample_size', '-n', type=int, default=5,
                        help='the sample size if a .trees file is not given')
    parser.add_argument('--effective_population_size', '-Ne', type=float,
                        default=5000, help='the effective population '
                        'size if a .trees file is not given')
    parser.add_argument('--iterations', type=int, default=10,
                        help= 'number of mcmc iterations')
    parser.add_argument('--sample_step', type=int, default= 5,
                        help=' the MCMC sample steps for storing the outputs')
    parser.add_argument('--sequence_length', '-l', type=float, default=55000,
                        help='the sequence length if a .trees file is not given')
    parser.add_argument('--recombination_rate', '-rho', type=float, default=1e-8,
                        help='the recombination rate ')
    parser.add_argument('--mutation_rate', '-mu', type=float, default=1e-8,
                        help='the mutation rate ')
    parser.add_argument('--random_seed', '-seed', type=int, default=1234,
                        help='a random seed for msprime & AW simulation')
    parser.add_argument('--outputdir', nargs="?", default=None,
                        help='the directory in which to store the intermediate files.'
                        ' If None, files are saved under temporary names')
    parser.add_argument('--full_prefix', type=str, default="out",
                        help='The full prefix of the out path')
    parser.add_argument( '-p','--plot', help="plot the output", action="store_true")
    parser.add_argument('--verbosity', '-v', action='count', default=0)
    args = parser.parse_args()
    log_level = logging.WARNING
    if args.verbosity == 1:
        log_level = logging.INFO
    if args.verbosity >= 2:
        log_level = logging.DEBUG
    logging.basicConfig(
        format='%(asctime)s %(message)s', level=log_level, stream=sys.stdout)

    if args.trees_file is None:
        logging.info("Running a new simulation with n {}, Ne {}, l {}, rho {}, mu {}".format(
        args.sample_size, args.effective_population_size, args.sequence_length,
        args.recombination_rate, args.mutation_rate))
        ts = msprime.simulate(
            sample_size = args.sample_size,
            Ne=args.effective_population_size,
            length=args.sequence_length,
            recombination_rate=args.recombination_rate,
            mutation_rate=args.mutation_rate,
            random_seed=args.random_seed)
    else:
        logging.warning("Loading a user-specified simulation file: WARNING, "
                        "argweaver may end up being run with"
                        " different parameters from the simulation")
    if args.outputdir == None:
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as aw_out_dir:
            logging.info("Saving everything to temporary files"
                         " (temporarily stored in {})".format(aw_out_dir))
            args.outputdir = aw_out_dir
            if args.trees_file is None:
                args.trees_file = os.path.join(aw_out_dir, "sim.trees")
                ts.dump(args.trees_file, zlib_compression=True)
            main(args)
    else:
        if not os.path.isdir(args.outputdir):
            logging.info("Output dir {} does not exist: creating it".format(args.outputdir))
            os.mkdir(args.outputdir)
        if len(os.listdir(args.outputdir)) > 0:
            logging.info("Output dir {} already contains files: deleting them".format(args.outputdir))
            import shutil
            shutil.rmtree(args.outputdir)
            os.mkdir(args.outputdir)
        if args.trees_file is None:
            args.trees_file = os.path.join(args.outputdir, "sim.trees")
            ts.dump(args.trees_file, zlib_compression=True)
        else:
            args.trees_file = args.trees_file.name
        main(args)

if __name__ == "__main__":
    aw_infer()

