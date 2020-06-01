from plot import *

'''
python  file_management.py --replicate 162 \
    --general_path "/data/projects/punim0594/Ali/phd/mcmc_out/aw/r4/n10L100K" \
    --read_summary_mf 
'''

def read_summary_multiple_folder(replicate, general_path, argweaver = False):
    '''read the summary of the posterior samples which are in
    multiple folders, output the mean of samples in a pandas_df
     '''
    if not argweaver:
        summary_all = pd.DataFrame(columns=('lower likelihood','likelihood','upper likelihood',"std likelihood",
                                            'lower prior', 'prior','upper prior',"std prior",
                                            'lower posterior', "posterior","upper posterior",'std posterior',
                                            'lower ancestral recomb','ancestral recomb',
                                            "upper ancestral recomb",'std ancestral recomb',
                                            'lower non ancestral recomb', 'non ancestral recomb',
                                            'upper non ancestral recomb','std non ancestral recomb',
                                            'lower branch length','branch length',
                                            'upper branch length','std branch length',
                                            'lower total recomb','total recomb','upper total recomb',
                                            'std total recomb'))
        for i in range(replicate):
            out_path = general_path+"/out" +str(i)
            print("/out" +str(i))
            if os.path.isfile(out_path+"/summary.h5"):
                f = Figure(out_path)
                df= f.data
                df_mean = df.mean(axis = 0).values.tolist()
                df_025_pecentage = df.quantile([0.025]).values.tolist()[0]
                df_975_pecentage = df.quantile([0.975]).values.tolist()[0]
                df_std = df.std(axis=0).values.tolist()
                new_row=[]
                for item in range(6):
                    new_row.extend([df_025_pecentage[item], df_mean[item],df_975_pecentage[item], df_std[item]])
                # for total rec
                df['total recomb'] = df.apply(lambda x: x['ancestral recomb'] + x['non ancestral recomb'], axis=1)
                new_row.extend([df['total recomb'].quantile([0.025]).tolist()[0],
                                df['total recomb'].mean(axis = 0).tolist(),
                                df['total recomb'].quantile([0.975]).tolist()[0],
                                df['total recomb'].std(axis = 0).tolist()])
                summary_all.loc[0 if math.isnan(summary_all.index.max())\
                    else summary_all.index.max() + 1] =  new_row
            else:
                summary_all.loc[0 if math.isnan(summary_all.index.max())\
                        else summary_all.index.max() + 1] =[None for i in range(28)]
    else:
        summary_all = pd.DataFrame(columns=['lower prior', 'prior','upper prior',"std prior",
                                            'lower likelihood','likelihood','upper likelihood',"std likelihood",
                                            'lower posterior', "posterior","upper posterior",'std posterior',
                                            'lower total recomb','total recomb','upper total recomb',
                                            'std total recomb',
                                            'lower branch length', 'branch length',
                                            'upper branch length', 'std branch length'])
        for i in range(replicate):
            out_path = general_path+"/out" +str(i)
            print("/out" +str(i))
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
                new_row=[]
                for item in range(5):
                    new_row.extend([df_025_pecentage[item], df_mean[item],df_975_pecentage[item], df_std[item]])
                summary_all.loc[0 if math.isnan(summary_all.index.max())\
                        else summary_all.index.max() + 1] =  new_row
            else:
                summary_all.loc[0 if math.isnan(summary_all.index.max())\
                        else summary_all.index.max() + 1] =[None for i in range(20)]
    #save true df
    summary_all.to_hdf(general_path + "/summary_all.h5", key = "df")
    print(summary_all)
    print("DONE")

def main(args):
    if args.read_summary_mf:
        read_summary_multiple_folder(replicate=args.replicate,
                        general_path = args.general_path,
                        argweaver = args.argweaver)
if __name__=='__main__':
    import argparse
    import os
    parser = argparse.ArgumentParser(prog="sim_ts",description='simulate datasets from ts')
    parser.add_argument('--replicate', type = int,default=10,help='number of data sets for each ratio')
    parser.add_argument('--general_path', '-O',type=str, default=os.getcwd(), help='The output path')
    parser.add_argument( "--argweaver", help="manage the argweaver output, otherwise arginfer", action="store_true")
    parser.add_argument( "--read_summary_mf", help="read_summary_multiple_folder function", action="store_true")
    args = parser.parse_args()
    main(args)
