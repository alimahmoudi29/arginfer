import msprime
import tskit
import pandas as pd
import shutil
import random
import numpy as np

'''
simulate data sets with different mu/r ratios from msprime

ex: 
python3 simulate.py --replicate 3 --ratios 1  \
 --mu 1e-8 --Ne 5000 -n 5 -L 1e3  \
--out_path /Users/amahmoudi/Ali/phd/github_projects/mcmc/test1/ts_sim
'''

def get_ts_seq(ts):
    '''
    return the sequences simulated by msprime
    '''
    seqs=ts.genotype_matrix()
    seqs= pd.DataFrame(seqs)
    return(seqs)

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
    #------- mkdir
    if not os.path.exists(args.out_path):
            os.makedirs(args.out_path)
    else:# if exists, first delete it and then create the directory
        shutil.rmtree(args.out_path)
        os.mkdir(args.out_path)
    args.ratios = list(args.ratios)
    for ratio in args.ratios:
        ratio = int(ratio)
        simulate_ts(replicate= args.replicate, ratio= ratio,
                    mut_rate = args.mutation_rate, Ne= args.Ne,
                    sample_size= args.sample_size,
                    length= args.length,
                    out_path= args.out_path+"/sim_r"+str(ratio))
    print("==========DONE")

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
    args = parser.parse_args()
    main(args)




















