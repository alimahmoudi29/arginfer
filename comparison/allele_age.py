'''
compute the allele age for all the data sets
(or for all the sampled ARGs in ne data set)
'''

'''
python allele_age.py --general_path /data/projects/punim0594/Ali/phd/mcmc_out/ARGinfer/M2/n10L100K_r4/out5 \
 --seq_length 1e5 \
 -d   /home/amahmoudi/miniconda3/envs/py27/bin \
 --arginfer_allele_age
 --argweaver_allele_age
'''

import os
import pandas as pd
import sys
from sortedcontainers import SortedSet
import numpy as np
import subprocess
f_dir = os.path.dirname(os.getcwd())#+"/ARGinfer"
sys.path.append(f_dir)
# print(sys.path)
import argbook

def arginfer_allele_age(general_path):
    '''
    allele age for the mcmc samples for one data set: say we have n sampled ARGs,
    each arg has its own allele age for each allele. For, each ARG, we would have a
    dataframe with columns [sites, recent age, mid age, latest age]. Now we need to average
    over them for all n ARG to have a single df. But we also want the 95% and 50% intervals
    so, we need to find them for "mid age" all all the ARG and add to final df.

    :return a df with colimns [site, recent age, mid age, latest age, lower age,
     upper age, lower25 age, upper75 age, std age].
    '''
    allele_df= pd.DataFrame(columns=["site", "recent age", "mid age", "latest age"])
    temporary_df= pd.DataFrame()
    count = 0
    for entry in os.scandir(general_path):
        if (entry.path.endswith(".arg")) and entry.is_file():
            # print("ARG number:", count)
            arg = argbook.ARG().load(entry.path)
            single_age_df = arg.allele_age()
            temporary_df["data"+str(count)] = single_age_df["mid age"]
            if count == 0:
                allele_df = single_age_df
            else:
                allele_df = allele_df + single_age_df
            count += 1
    # average over the allele age
    print("count is ", count)
    allele_df /= count
    # now add the CI to the allele df
    allele_df["lower age"] = temporary_df.quantile([0.025], axis=1).values[0]
    allele_df["upper age"] = temporary_df.quantile([0.975], axis =1).values[0]
    allele_df["lower25 age"] = temporary_df.quantile([0.25], axis=1).values[0]
    allele_df["upper75 age"] = temporary_df.quantile([0.75], axis =1).values[0]
    allele_df["std age"] = temporary_df.std(axis = 1)
    allele_df.to_hdf(general_path + "/allele_age.h5", key = "df")
    print("-------DONE!")

def main(args):
    if args.arginfer_allele_age:
        arginfer_allele_age(general_path = args.general_path) #sequence_length= args.seq_length
    # if args.argweaver_allele_age:
    #     argweaver_allele_age(args.general_path, args.ARGweaver_executable_dir,
    #                     sequence_length= args.seq_length)

if __name__=='__main__':
    import argparse
    import os
    parser = argparse.ArgumentParser(prog="",description='allele for the mcmc samples of one data set')
    parser.add_argument('--general_path', '-O',type=str, default=os.getcwd(), help='The output path')
    parser.add_argument('--seq_length','-L', type=float, default=1e4,help='sequence length')
    parser.add_argument('--ARGweaver_executable_dir', '-d',
        default= "/Users/amahmoudi/Dropbox/PhDMelbourne/Paper/ARGweaver/argweaver-master/bin/",
                        #os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','argweaver/bin/')
        help='the path to the directory containing the ARGweaver executables')
    parser.add_argument( "--arginfer_allele_age", help="ARGinfer allele age", action="store_true")
    parser.add_argument( "--argweaver_allele_age", help="argweaver allele age", action="store_true")
    args = parser.parse_args()
    main(args)

