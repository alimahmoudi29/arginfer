'''responsible for tmrca of an ARG and mcmc samples '''
import os
import pandas as pd
import sys
import subprocess
f_dir = os.path.dirname(os.getcwd())#+"/ARGinfer"
sys.path.append(f_dir)
# print(sys.path)
import argbook

'''
python tmrca.py --general_path '/data/projects/punim0594/Ali/phd/mcmc_out/ARGinfer/n10L100K_r4/out5' \
 --seq_length 1e5 \
 -d   /home/amahmoudi/miniconda3/envs/py27/bin \
 --arginfer_tmrca
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
            print("count:", count)
            arg = argbook.ARG().load(entry.path)
            tmrca_df["data"+str(count)] = arg.total_tmrca(sequence_length)
            count += 1
    # now get the summaries out
    summary_df = pd.DataFrame()
    summary_df["tmrca"] = tmrca_df.mean(axis = 1)
    summary_df["lower tmrca"] = tmrca_df.quantile([0.025], axis=1).values[0]
    summary_df["upper tmrca"] = tmrca_df.quantile([0.975], axis =1).values[0]
    summary_df["std tmrca"] = tmrca_df.std(axis = 1)
    summary_df.to_hdf(general_path + "/tmrca.h5", key = "df")

def argweaver_tmrca(general_path, ARGweaver_executable_dir):
    '''not efficient'''
    cmd = os.path.join(ARGweaver_executable_dir, 'arg-extract-tmrca ') +\
        general_path +'/out.%d.smc.gz ' +\
        ' > '+ general_path + '/tmrca.txt'
    subprocess.check_call(cmd, shell=True)
    # convert to dataframe similar as arginfer tmrca
    tmrca_df = pd.DataFrame(columns=("tmrca", "lower tmrca", "upper tmrca"))
    print("txt file written -------")
    with open(general_path + '/tmrca.txt')as f:
        for line in f:
            L = line.strip().split()[1:]
            for row in range(int(L[0]), int(L[1])):
                tmrca_df.loc[row] = [float(L[2]), float(L[3]), float(L[4])]
    tmrca_df.to_hdf(general_path + "/tmrca.h5", key = "df")

def main(args):
    if args.arginfer_tmrca:
        arginfer_tmrca(general_path = args.general_path,
                          sequence_length= args.seq_length)
    if args.argweaver_tmrca:
        argweaver_tmrca(args.general_path, args.ARGweaver_executable_dir)

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

