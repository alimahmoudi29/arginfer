import msprime
import tskit
import pandas as pd
import shutil
import os
import sys
import random
import numpy as np
f_dir = os.path.dirname(os.getcwd())#+"/ARGinfer"
sys.path.append(f_dir)
# print(sys.path)
import treeSequence
import math
from tqdm import tqdm
'''
simulate data sets with different mu/r ratios from msprime

ex: 
python3 simulate.py --replicate 3 --ratios 1  \
 --mu 1e-8 --Ne 5000 -n 5 -L 1e3 --generate --summary --tmrca \
--out_path /Users/amahmoudi/Ali/phd/github_projects/mcmc/test1/ts_sim
'''

def get_true_tmrca(tsfull):
    ts = tskit.TreeSequence.simplify(tsfull)
    tmrca = np.zeros(int(ts.sequence_length))
    for tree in ts.trees():
        tmrca[int(tree.interval[0]):int(tree.interval[1])]= tree.time(tree.root)
    return tmrca#wanted_tmrcas

def get_true_features(ts_full, ratio, mut_rate, Ne, true_df):
    '''get  likelihood, prior, posterior,
    numancestral_rec, nonancestral_Rec,
    total_rec, total_branch_Rec
    and add to a dataframe
    '''
    try:
        recomb_rate = float(mut_rate/ratio)
        tsarg = treeSequence.TreeSeq(ts_full)

        tsarg.ts_to_argnode()
        data = treeSequence.get_arg_genotype(ts_full)
        #----- true values
        arg = tsarg.arg
        log_lk = arg.log_likelihood(mut_rate, data)
        log_prior = arg.log_prior(ts_full.sample_size, ts_full.sequence_length,
                                            recomb_rate, Ne, False)
        true_df.loc[0 if math.isnan(true_df.index.max())\
                    else true_df.index.max() + 1] = [log_lk, log_prior, log_lk+log_prior,
                                                    arg.num_ancestral_recomb,
                                                    arg.num_nonancestral_recomb,
                                                    arg.num_ancestral_recomb+ arg.num_nonancestral_recomb,
                                                    arg.branch_length]
    except:
        true_df.loc[0 if math.isnan(true_df.index.max())\
                else true_df.index.max() + 1] = [None for i in range(7)]
    return true_df

def simulate_ts(replicate, ratio, mut_rate, Ne, sample_size,
                length, out_path):
    '''
    replicate: number of data sets to simulate
    to simulate and store ts in the given "out_path.
    rerturn: store the ts_arg, ts.
    '''
    recomb_rate = float(mut_rate/ratio)
    if not os.path.exists(out_path):
            os.makedirs(out_path)
    else:
        shutil.rmtree(out_path)
        os.mkdir(out_path)
    for i in range(replicate):
        ts_arg = msprime.simulate(sample_size= sample_size, Ne= Ne,
                                  length=length,
                    recombination_rate=recomb_rate, mutation_rate= mut_rate,
                                  record_full_arg=True)
        #--- save the simulated arg for calculating the true values
        name = "n"+str(sample_size)+"Ne"+str(int(Ne/1000))+"K_L"+ \
               str(int(length/1000))+"K"+ "_iter"+ str(i)
        #save full arg
        ts_arg.dump(out_path+"/" + name+".args")

def main(args):
    '''
    :param replicate: the number of replicates for datas with same ratio
    :param ratios: a list of mut/recomb ratios [1,2,4,6]
    :param mut_rate:
    :param Ne: population size
    :param sample_size:
    :param length: seq len
    :param random_seed:
    :param record_full_arg:  record full arg in the msprime
    :param out_path:
    :return:
    '''
    if args.generate:
        #------- mkdir
        if not os.path.exists(args.out_path):
                os.makedirs(args.out_path)
        else:# if exists, first delete it and then create the directory
            shutil.rmtree(args.out_path)
            os.mkdir(args.out_path)
        args.ratios = list(args.ratios)
        for ratio in args.ratios:
            ratio = float(ratio)
            if ratio == int(ratio):
                ratio = int(ratio)
            simulate_ts(replicate= args.replicate, ratio= ratio,
                        mut_rate = args.mutation_rate, Ne= args.Ne,
                        sample_size= args.sample_size,
                        length= args.length,
                        out_path= args.out_path+"/sim_r"+str(ratio))
    if args.summary:
        true_df = pd.DataFrame(columns=('likelihood', 'prior', "posterior",
                                             'ancestral recomb', 'non ancestral recomb',
                                                'total recomb', 'branch length'))
        if not args.generate:
            ratio = list(args.ratios)[0]
            ratio = float(ratio)
            if ratio == int(ratio):
                ratio = int(ratio)
        out_path = args.out_path+"/sim_r"+str(ratio)
        for i in tqdm(range(args.replicate), ncols=100, ascii=False):
        # for i in range(args.replicate):
            name = "n"+str(args.sample_size)+"Ne"+str(int(args.Ne/1000))+"K_L"+ \
                   str(int(args.length/1000))+"K"+ "_iter"+ str(i)
            ts_full = msprime.load(out_path +"/"+name+".args")
            true_df = get_true_features(ts_full, ratio, args.mutation_rate, args.Ne, true_df)
            if args.tmrca:# tmrca
                tmrca = get_true_tmrca(ts_full)
                np.save(out_path + '/true_tmrca'+str(i)+'.npy', tmrca)
        #save true df
        true_df.to_hdf(out_path + "/true_summary.h5", key = "df")

if __name__=='__main__':
    import argparse
    import os
    parser = argparse.ArgumentParser(prog="sim_ts",description='simulate datasets from ts')
    parser.add_argument('--replicate', type = int,default=10,help='number of data sets for each ratio')
    parser.add_argument('--mutation_rate', '-mu', type=float, default=1e-8, help='the mutation rate ')
    parser.add_argument('--Ne', type=int, default= 5000, help=' The population size')
    parser.add_argument('--sample_size','-n',type=int, default= 5, help='the sample size')
    parser.add_argument('--ratios', nargs='+',  default= 1, help= 'an array of the mutation/recomb rate ratios')
    parser.add_argument('--length', '-L',type=float, default= 1e4, help=' The sequence length')
    parser.add_argument('--out_path', '-O',type=str, default=os.getcwd(), help='The output path')
    parser.add_argument( "--generate", help="if we want to simulate new ts", action="store_true")
    parser.add_argument( "--summary", help="if we want to draw the summary of existing ts", action="store_true")
    parser.add_argument( "--tmrca", help="if we need tmrca", action="store_true")
    args = parser.parse_args()
    main(args)




















