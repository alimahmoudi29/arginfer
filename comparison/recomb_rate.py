'''estimate local recombination rate along the genome.
--------
Note: this is different from the overal recombination rate
that is estimated for the whole genome as a pack!: recrate_boxplot in plot.py
----------
'''
import os
import pandas as pd
import sys
from sortedcontainers import SortedSet
import numpy as np
import subprocess
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import math
from scipy import stats
import seaborn as sns
from tqdm import tqdm
f_dir = os.path.dirname(os.getcwd())#+"/ARGinfer"
sys.path.append(f_dir)
# print(sys.path)
import argbook

'''Usage:
/home/amahmoudi/miniconda3/envs/py37/bin/python \
 recomb_rate.py -d   /home/amahmoudi/miniconda3/envs/py27/bin \
 --weaver_general_path /data/projects/punim0594/Ali/phd/mcmc_out/aw/r1/n10L100K/out5 \
 --infer_general_path /data/projects/punim0594/Ali/phd/mcmc_out/ARGinfer/M2/n10L100K_r1/out5
  --window 10000 \ 

 '''

def arginfer_recomb_rate(general_path):
    '''
    1. calculate tree branch lengths in all sampled ARGs
    2. average over the branch lengths for each interval between consecutive recomb pos
    3. order all recomb pos in one array and then start to calculate rec rate for a sliding window
        first interval is (min(recomb pos), min_recomb_pos + window), then second interval
        is the second min recomb pos , ...
    TODO: complete thsi. After it is done, the plot can be
        smoothed using moving average, similar to calculateLD.py
    '''

def argweaver_recomb_rate(general_path, ARGweaver_executable_dir, window, ntimes):
    '''
    ARGweaver: https://github.com/mdrasmus/argweaver/blob/master/bin/arg-extract-recomb
    Problem: not clear how window works in ARGweaver
    So, I modified to return only the recom positions for all the ARG samples for one data set:
    in /home/amahmoudi/miniconda3/envs/py27/bin/arg-extract-breaks, I added:
    util.print_row(item["pos"]) in line 44 and commented the last two lines (84, 85):
    util.print_row(chrom,..., stats.percentile(vals, 0.975))

    :param general_path: path to the ARGs
    :param ARGweaver_executable_dir: ARGweaver executable directory
    :param window Window over which to average number of break points
            (see the webpage above): since I changed the code to get rec pos,
             this is not important anymore.
    :return: all the breakpoints
    '''
    cmd = os.path.join(ARGweaver_executable_dir, 'arg-extract-recomb ') +\
        general_path +'/out.%d.smc.gz ' +\
        " --window "+ str(window) +\
        " --ntimes "+ str(ntimes) #+\
        #' > '+ general_path + '/recomb_breaks.txt' # write
    subprocess.check_call(cmd, shell=True)
    # br = np.loadtxt(general_path + '/recomb_breaks.txt', delimiter="\t")
    # return br

def main(args):
    # if args.infer_recomb_breaks:
    # infer_br= arginfer_recomb_breaks(general_path = args.infer_general_path) #sequence_length= args.seq_length
    # if args.weaver_recomb_breaks:
    argweaver_recomb_rate(args.weaver_general_path, args.ARGweaver_executable_dir,
                              args.window, ntimes=args.ntimes)
    #now lets plot the histogram for them
    # plt.figure()#figsize=[10,8]
    # hist,bin_edges = np.histogram(infer_br, bins=2000, density= True)
    # plt.bar(bin_edges[:-1], hist, width = 50, color='#0504aa',alpha=0.9, label="ARGinfer")
    # plt.xlim(min(bin_edges), max(bin_edges))
    # # plt.grid(axis='y', alpha=0.75)
    #
    # hist,bin_edges = np.histogram(weaver_br, bins=2000, density= True)
    # plt.bar(bin_edges[:-1], hist, width = 50, color='#b22222',alpha=0.7, label="ARGweaver")
    # plt.xlim(min(bin_edges), max(bin_edges))
    # # plt.grid(axis='y', alpha=0.75)
    # plt.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
    # plt.ticklabel_format(style='sci',scilimits=(0,0),axis='x')
    # plt.xlabel('Genomic position',fontsize=9)
    # plt.xticks(fontsize=9)
    # plt.yticks(fontsize=9)
    # plt.ylabel('Recombination density',fontsize=9)
    # plt.legend()
    # # plt.title('Normal Distribution Histogram',fontsize=15)
    # figure_name= "recombination_count"
    # plt.savefig(args.infer_general_path+"/{}.pdf".format(figure_name),
    #             bbox_inches='tight', dpi=400)
    # plt.close()
    # plt.show()

if __name__=='__main__':
    import argparse
    import os
    parser = argparse.ArgumentParser(prog="",description='local recombination rate along the genome in blocks'
                                                         'of seq_length/window ')
    parser.add_argument('--infer_general_path', '-O',type=str, default=os.getcwd(), help='ARGinfer output path ')
    parser.add_argument('--weaver_general_path', type=str, default=os.getcwd(), help='ARGweaver output path')
    parser.add_argument('--ARGweaver_executable_dir', '-d',
        default= "/Users/amahmoudi/Dropbox/PhDMelbourne/Paper/ARGweaver/argweaver-master/bin/",
                        #os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','argweaver/bin/')
        help='the path to the directory containing the ARGweaver executables')
    # parser.add_argument( "--infer_recomb_breaks", help="ARGinfer recombi positions", action="store_true")
    # parser.add_argument( "--weaver_recomb_breaks", help="argweaver recomb positions", action="store_true")
    parser.add_argument('--window', type=int, default=10000,
                        help= 'Window over which to average number of break points : this is not used for now')
    parser.add_argument('--ntimes', type=int, default=20,
                        help= 'number of event times in argweaver')
    args = parser.parse_args()
    main(args)
