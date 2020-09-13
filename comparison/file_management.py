from plot import *
from tqdm import tqdm
'''
after ARGinfer, or ARGweaver done for multiple data sets!
 this is used to summarise the info of 
all the data sets in a df file called summary_all.h5. In fact, 
this file is 
a summary of all the individual summaries for each data set:

For ARGinfer: 
python  file_management.py --replicate 162 \
    --general_path "/data/projects/punim0594/Ali/phd/mcmc_out/ARGinfer/M2/n10L100K_r1" \
    --read_summary_mf 
    
#--------------------------------------
For ARGweaver:
python  file_management.py --replicate 162 \
    --general_path "/data/projects/punim0594/Ali/phd/mcmc_out/aw/r2/n10L100K" \
    --read_summary_mf --argweaver
'''

def read_summary_multiple_folder(replicate, general_path,
                                 argweaver = False):
    '''read the summary of the posterior samples which are in
    multiple folders, output the mean of samples in a pandas_df
     '''
    if not argweaver:
        summary_all = pd.DataFrame(columns=('lower likelihood','likelihood','upper likelihood',"std likelihood",
                                            "lower25 likelihood", "upper75 likelihood",
                                            'lower prior', 'prior','upper prior',"std prior",
                                            "lower25 prior", "upper75 prior",
                                            'lower posterior', "posterior","upper posterior",'std posterior',
                                            "lower25 posterior", "upper75 posterior",
                                            'lower ancestral recomb','ancestral recomb',
                                            "upper ancestral recomb",'std ancestral recomb',
                                            "lower25 ancestral recomb", "upper75 ancestral recomb",
                                            'lower non ancestral recomb', 'non ancestral recomb',
                                            'upper non ancestral recomb','std non ancestral recomb',
                                            "lower25 non ancestral recomb", "upper75 non ancestral recomb",
                                            'lower branch length','branch length',
                                            'upper branch length','std branch length',
                                            "lower25 branch length", "upper75 branch length",
                                            'lower total recomb','total recomb','upper total recomb',
                                            'std total recomb',"lower25 total recomb", "upper75 total recomb" ))
        for i in tqdm(range(replicate), ncols=100, ascii=False):
        # for i in range(replicate):
            out_path = general_path+"/out" +str(i)
            if os.path.isfile(out_path+"/summary.h5"):
                f = Figure(out_path)
                df= f.data
                df_mean = df.mean(axis = 0).values.tolist()
                df_025_pecentage = df.quantile([0.025]).values.tolist()[0]
                df_975_pecentage = df.quantile([0.975]).values.tolist()[0]
                df_std = df.std(axis=0).values.tolist()
                df_25_pecentage = df.quantile([0.25]).values.tolist()[0]
                df_75_pecentage = df.quantile([0.75]).values.tolist()[0]
                new_row=[]
                for item in range(6):
                    new_row.extend([df_025_pecentage[item], df_mean[item],df_975_pecentage[item], df_std[item],
                                    df_25_pecentage[item], df_75_pecentage[item]])
                # for total rec
                df['total recomb'] = df.apply(lambda x: x['ancestral recomb'] + x['non ancestral recomb'], axis=1)
                new_row.extend([df['total recomb'].quantile([0.025]).tolist()[0],
                                df['total recomb'].mean(axis = 0).tolist(),
                                df['total recomb'].quantile([0.975]).tolist()[0],
                                df['total recomb'].std(axis = 0).tolist(),
                                df['total recomb'].quantile([0.25]).tolist()[0],
                                df['total recomb'].quantile([0.75]).tolist()[0]])
                summary_all.loc[0 if math.isnan(summary_all.index.max())\
                    else summary_all.index.max() + 1] =  new_row
            else:
                summary_all.loc[0 if math.isnan(summary_all.index.max())\
                        else summary_all.index.max() + 1] =[None for i in range(42)]
    else:
        summary_all = pd.DataFrame(columns=['lower prior', 'prior','upper prior',"std prior",
                                            "lower25 prior", "upper75 prior",
                                            'lower likelihood','likelihood','upper likelihood',"std likelihood",
                                            "lower25 likelihood", "upper75 likelihood",
                                            'lower posterior', "posterior","upper posterior",'std posterior',
                                            "lower25 posterior", "upper75 posterior",
                                            'lower total recomb','total recomb','upper total recomb',
                                            'std total recomb',
                                            "lower25 total recomb", "upper75 total recomb",
                                            'lower branch length', 'branch length',
                                            'upper branch length', 'std branch length',
                                            "lower25 branch length", "upper75 branch length"])
        for i in tqdm(range(replicate), ncols=100, ascii=False):
        # for i in range(replicate):
            out_path = general_path+"/out" +str(i)
            if os.path.isfile(out_path+"/out.stats"):
                f= Figure(out_path, argweaver = argweaver)
                stats_df = f.data
                del stats_df['noncompats']
                burn_in=10000
                sample_step= 20
                # keep every sample_stepth after burn_in
                stats_df = stats_df.iloc[burn_in::sample_step, :]
                df_mean = stats_df.mean(axis = 0).values.tolist()
                df_025_pecentage = stats_df.quantile([0.025]).values.tolist()[0]
                df_975_pecentage = stats_df.quantile([0.975]).values.tolist()[0]
                df_std = stats_df.std(axis=0).values.tolist()
                df_25_pecentage = stats_df.quantile([0.25]).values.tolist()[0]
                df_75_pecentage = stats_df.quantile([0.75]).values.tolist()[0]
                new_row=[]
                for item in range(5):
                    new_row.extend([df_025_pecentage[item], df_mean[item],df_975_pecentage[item], df_std[item],
                                    df_25_pecentage[item], df_75_pecentage[item]])
                summary_all.loc[0 if math.isnan(summary_all.index.max())\
                        else summary_all.index.max() + 1] =  new_row
            else:
                summary_all.loc[0 if math.isnan(summary_all.index.max())\
                        else summary_all.index.max() + 1] =[None for i in range(30)]
    #save true df
    summary_all.to_hdf(general_path + "/summary_all.h5", key = "df")
    print(summary_all.head(10))
    print("DONE")

def main(args):
    if args.read_summary_mf:
        read_summary_multiple_folder(replicate=args.replicate,
                        general_path = args.general_path,
                        argweaver = args.argweaver)
if __name__=='__main__':
    import argparse
    import os
    parser = argparse.ArgumentParser(prog="",description='')
    parser.add_argument('--replicate', type = int,default=10,help='number of data sets for each ratio')
    parser.add_argument('--general_path', '-O',type=str, default=os.getcwd(), help='The output path')
    parser.add_argument( "--argweaver", help="manage the argweaver output, otherwise arginfer", action="store_true")
    parser.add_argument( "--read_summary_mf", help="read_summary_multiple_folder function", action="store_true")
    args = parser.parse_args()
    main(args)

