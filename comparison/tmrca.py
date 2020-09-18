'''responsible for tmrca of an ARG and mcmc samples '''
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

'''
python tmrca.py --general_path '/data/projects/punim0594/Ali/phd/mcmc_out/aw/r2/n10L100K/out10' \
 --seq_length 1e5 \
 -d   /home/amahmoudi/miniconda3/envs/py27/bin \
 --argweaver_tmrca
 --arginfer_tmrca
 
 #---------------
 r4: out5, 7
 r2: out2, 10
 r1: out2, 5, 15
 
'''

'''
 in bash for multiple data sets: 
 for i in {0..161}; 
 do /home/amahmoudi/miniconda3/envs/py37/bin/python tmrca.py 
 --general_path /data/projects/punim0594/Ali/phd/mcmc_out/aw/r1/n10L100K/out$i 
 --seq_length 1e5 -d   /home/amahmoudi/miniconda3/envs/py27/bin --argweaver_tmrca; done
'''



def arginfer_tmrca(general_path, sequence_length):
    '''
    tmrca for the mcmc samples of one data set: say we have n sampled ARGs and
    would like to average over the tmrca of all n ARGs for all sites
    and return the mean_tmrca, and the 95% credible interval and the standard deviation
    '''
    tmrca_df= pd.DataFrame()#[L, n] where n is the number of mcmc samples
    count = 0
    for entry in os.scandir(general_path):
        if (entry.path.endswith(".arg")) and entry.is_file():
            # print("ARG number:", count)
            arg = argbook.ARG().load(entry.path)
            tmrca_df["data"+str(count)] = arg.total_tmrca(sequence_length)
            count += 1
    # now get the summaries out
    summary_df = pd.DataFrame()
    summary_df["tmrca"] = tmrca_df.mean(axis = 1)
    summary_df["lower tmrca"] = tmrca_df.quantile([0.025], axis=1).values[0]
    summary_df["upper tmrca"] = tmrca_df.quantile([0.975], axis =1).values[0]
    summary_df["lower25 tmrca"] = tmrca_df.quantile([0.25], axis=1).values[0]
    summary_df["upper75 tmrca"] = tmrca_df.quantile([0.75], axis =1).values[0]
    summary_df["std tmrca"] = tmrca_df.std(axis = 1)
    summary_df.to_hdf(general_path + "/tmrca.h5", key = "df")
    print("-------DONE!")

def argweaver_tmrca(general_path, ARGweaver_executable_dir, sequence_length):
    '''
    ARGweaver: https://github.com/mdrasmus/argweaver/blob/master/bin/arg-extract-tmrca

    Note: since argweaver does not report 50%CI, I have added this to arg-extract-tmrca
        in /home/amahmoudi/miniconda3/envs/py27/bin to report 25 and 75 quantiles.

    tmrca for a data set: for each site, average over
    the tmrca given by the ARGs'''
    cmd = os.path.join(ARGweaver_executable_dir, 'arg-extract-tmrca ') +\
        general_path +'/out.%d.smc.gz ' +\
        ' > '+ general_path + '/tmrca.txt'
    subprocess.check_call(cmd, shell=True)
    # print("txt file written -------")
    df=  pd.read_csv(general_path + '/tmrca.txt', sep="\t", header=None)
    # print("read txt and convert to pandas df -----")
    df.columns = ["chr","start", "end", "tmrca", "lower tmrca", "upper tmrca",
                     "lower25 tmrca", "upper75 tmrca"]
    #--------- convert the df to have a row for each site
    tmrca_df = pd.DataFrame(columns=("tmrca", "lower tmrca", "upper tmrca",
                                     "lower25 tmrca", "upper75 tmrca"))
    break_points=SortedSet(df["end"])
    break_points.add(0)
    tm_np = np.zeros(int(sequence_length)*5).reshape(5, int(sequence_length))
    count =0
    while count < len(break_points)-1:
        row= df.iloc[[count], [3, 4,5,6,7]].values.tolist()[0]
        for j in range(5):
            tm_np[j][break_points[count]:break_points[count+1]]=row[j]
        count += 1
    tmrca_df["tmrca"]= tm_np[0]; tmrca_df["lower tmrca"]= tm_np[1]
    tmrca_df["upper tmrca"]= tm_np[2]; tmrca_df["lower25 tmrca"]= tm_np[3]
    tmrca_df["upper75 tmrca"]= tm_np[4]
    assert (tmrca_df["tmrca"].all() <= tmrca_df["upper75 tmrca"].all())
    # for row in range(df.shape[0]):
    #     for ind in range(df.iloc[row, 1], df.iloc[row, 2]):
    #         tmrca_df.loc[ind] = df.iloc[[row], [3, 4,5,6,7]].values.tolist()[0]
    # # with open(general_path + '/tmrca.txt')as f:
    #     for line in f:
    #         L = line.strip().split()[1:]
    #         for row in range(int(L[0]), int(L[1])):
    #             tmrca_df.loc[row] = [float(L[2]), float(L[3]), float(L[4]), float(L[5]), float(L[6])]
    tmrca_df.to_hdf(general_path + "/tmrca.h5", key = "df")
    print("-------DONE!")

def main(args):
    if args.arginfer_tmrca:
        arginfer_tmrca(general_path = args.general_path,
                          sequence_length= args.seq_length)
    if args.argweaver_tmrca:
        argweaver_tmrca(args.general_path, args.ARGweaver_executable_dir,
                        sequence_length= args.seq_length)

if __name__=='__main__':
    import argparse
    import os
    parser = argparse.ArgumentParser(prog="",description='tmrca for the mcmc samples of a dataset')
    parser.add_argument('--general_path', '-O',type=str, default=os.getcwd(), help='The output path')
    parser.add_argument('--seq_length','-L', type=float, default=1e4,help='sequence length')
    parser.add_argument('--ARGweaver_executable_dir', '-d',
        default= "/Users/amahmoudi/Dropbox/PhDMelbourne/Paper/ARGweaver/argweaver-master/bin/",
                        #os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','argweaver/bin/')
        help='the path to the directory containing the ARGweaver executables')
    parser.add_argument( "--arginfer_tmrca", help="tmrca", action="store_true")
    parser.add_argument( "--argweaver_tmrca", help="argweaver tmrca", action="store_true")
    args = parser.parse_args()
    main(args)

