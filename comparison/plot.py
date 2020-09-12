import numpy as np
np.random.seed(19680801)
data = np.random.randn(2, 100)
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import math
from scipy import stats
import seaborn as sns

class Figure(object):
    """
    Superclass of figures . Each figure is a concrete subclass.
    """
    # name = None
    def __init__(self, outpath = os.getcwd() +"/output",
                 name = "summary", argweaver = False):
        self.outpath = outpath
        self.name= name
        if not argweaver:# arginfer
            datafile_name = self.outpath + "/{}.h5".format(self.name)
            self.data = pd.read_hdf(datafile_name, mode="r")
        else: # argweaver
            datafile_name = self.outpath+"/out.stats"
            self.data = self.argweaver_read(datafile_name)

    def save(self, figure_name=None, bbox_inches="tight"):
        if figure_name is None:
            figure_name = self.name
        print("Saving figure '{}'".format(figure_name))
        plt.savefig(self.outpath+"/{}.pdf".format(figure_name),
                    bbox_inches='tight', dpi=400)
        # plt.savefig("figures/{}.png".format(figure_name), bbox_inches='tight', dpi=400)
        plt.close()

    def load_true_values(self,filename = "true_values.npy"):
        data_filename = self.outpath + "/{}".format(filename)
        return np.load(data_filename)

    def argweaver_read(self, aw_out):
        '''takes the argweaver out stats file and read id as a pd.df'''
        aw_stat_df = pd.DataFrame(columns=["prior","likelihood","posterior",
                                         "total recomb","noncompats", "branch length"])
        with open(aw_out, "r") as f:
            for line in f:
                if line[0]=="r":
                    l= line.split("\t")
                    l.remove("resample")
                    l=[float(e) for e in l]
                    new_df= pd.DataFrame([l[1:]],
                                         columns = aw_stat_df.columns.values.tolist())
                    aw_stat_df= aw_stat_df.append(new_df, ignore_index = True)
        return aw_stat_df

class Trace(Figure):

    def arginfer_trace(self,  true_values= True):
        df = self.data
        truth =  self.load_true_values()
        true_branch_length = truth[3]
        true_anc_recomb= truth[4]
        true_nonanc_rec = truth[5]
        fig = plt.figure()
        fig.subplots_adjust(hspace = 0.35, wspace = 0.6)
        for i,  d in zip(range(4), ["posterior", "branch length",
                                    "ancestral recomb", "non ancestral recomb"]):
            fig.add_subplot(2, 2, i+1)
            df = self.data[d]
            plt.plot(df)
            plt.ticklabel_format(style='sci',scilimits=(0,0),axis='y')# (0,0) includes all
            if true_values:
                plt.axhline(y= truth[i+2], color="r", linestyle = "--", lw= 1)
            plt.ylabel(d)
            if i>1:
                plt.xlabel("Iteration")
        fig.suptitle("Iter = " +str(int(self.data.setup[0]/1000)) + "K "+", thin = "+\
            str(int(self.data.setup[1]))+ " "+", burn: "+ str(int(self.data.setup[2]))+\
            ", n= " + str(int(self.data.setup[3]))+", Ne = "+ str(int(self.data.setup[6]/1000))  +\
                     "K,\n L= "+ str(int(self.data.setup[4]/1000))+\
            "K, m= " + str(int(self.data.setup[5]))+ ", accept= "+ str(self.data.setup[9])+\
            ", CPU time = " + str(int(self.data.setup[10]/60))+ " min\n" +
                     "detail accept: ["+ str(self.data.setup[11]) +", " + str(self.data.setup[12])+ ", " +
                     str(self.data.setup[13]) + ", "+ str(self.data.setup[14])+ ", "
                     + str(self.data.setup[15])+ " ,"+ str(self.data.setup[16]) +\
                     " ,"+ str(self.data.setup[17])+"]")
        self.save(figure_name="arginfertrace" + time.strftime("%Y%m%d-%H%M%S"))
        plt.show()

    def argweaver_trace(self, true_values= True):
        df = self.data
        iterations= self.data.shape[0]
        cpu_time = np.load(self.outpath+"/out_time.npy")[0]
        if true_values:
            truth =  self.load_true_values()
            true_branch_length = truth[3]
            true_anc_recomb= truth[4]
            true_nonanc_rec = truth [5]

            wanted_true = [truth[2], true_branch_length,
                           true_anc_recomb+true_nonanc_rec]
        burn_in=10000
        sample_step=20
        # keep every sample_stepth after burn_in
        df = df.iloc[burn_in::sample_step, :]
        fig = plt.figure()
        fig.subplots_adjust(hspace = 0.35, wspace = 0.6)
        for i,  d in zip(range(3), ["posterior", "branch length",
                                    "total recomb"]):
            fig.add_subplot(2, 2, i+1)
            df = self.data[d]
            plt.plot(df)
            plt.ticklabel_format(style='sci',scilimits=(0,0),axis='y')# (0,0) includes all
            if true_values:
                if i != 0:# posterior is different formula in argweaver
                    plt.axhline(y= wanted_true[i], color="r", linestyle = "--", lw= 1)
                if i == 2:
                    plt.axhline(y= true_anc_recomb, color="g", linestyle = "--", lw= 1)
            plt.ylabel(d)
            if i>1:
                plt.xlabel("Iteration")
        fig.suptitle("Iter = " +str(int(iterations/1000)) + "K "+", thin = "+\
            str(int(sample_step))+ " "+", burn: "+ str(int(burn_in))+\
            ", CPU time = " + str(int(cpu_time/60))+ " min\n")

        self.save(figure_name="argweavertrace" + time.strftime("%Y%m%d-%H%M%S"))
        plt.show()

class Scatter(object):

    def __init__(self, truth_path, inferred_path,
                 columns = ["likelihood","prior", "posterior", "branch length"], std = False):
        self.columns = columns
        self.truth_path = truth_path
        self.inferred_path = inferred_path
        if len(self.columns)/2 > 1:
            plot_dimension =[math.ceil(len(self.columns)/2),2]
        elif len(self.columns)/2 == 1:# adjust size
            plot_dimension =[2,2]#math.ceil(len(self.columns)/2)
            # plt.rcParams["figure.figsize"] = (4,1.5)
        else:
            plot_dimension =[1, 1]
        self.fig = plt.figure(tight_layout=False)
        print("plot size in inch:",plt.rcParams.get('figure.figsize'))
        self.gs = gridspec.GridSpec(int(plot_dimension[0]), int(plot_dimension[1]))
        if not std:
            self.truth_data = pd.read_hdf(self.truth_path +  "/true_summary.h5", mode="r")
            self.inferred_data = pd.read_hdf(self.inferred_path + '/summary_all.h5', mode="r")
        else: # both inferred, note that instead of truth
            self.inferred_data = pd.read_hdf(self.inferred_path + '/summary_all.h5', mode="r")
            self.inferred2_data = pd.read_hdf(self.truth_path + "/summary_all.h5", mode="r")
        #row indexes with not None values
        self.not_None_rows = np.where(self.inferred_data['prior'].notnull())[0]

    def single_scatter(self, column_index, CI = False, argweaver= False, coverage =True):
        line_color= "red"
        point_color= "black"
        ecolor = "purple"
        elinewidth =0.9
        self.gs.update(wspace=0.3, hspace=0.7) # set the spacing between axes.
        if column_index <2:
            ax = self.fig.add_subplot(self.gs[0, column_index])
        elif column_index<4:
            ax = self.fig.add_subplot(self.gs[1, column_index-2])
        elif column_index <6:
            ax = self.fig.add_subplot(self.gs[2, column_index-4])
        else:
            ax = self.fig.add_subplot(self.gs[3, column_index-6])
        if argweaver and self.columns[column_index] == "ancestral recomb":
            col= 'total recomb'
        else:
            col = self.columns[column_index]
        if not CI:
            ax.errorbar(self.truth_data.loc[self.not_None_rows][self.columns[column_index]],
                            self.inferred_data.loc[self.not_None_rows][col],
                            color = point_color,linestyle='', fmt=".")#fmt="o"
        else:
            ax.errorbar(self.truth_data.loc[self.not_None_rows][self.columns[column_index]],
                        self.inferred_data.loc[self.not_None_rows][col],
                        yerr= [self.inferred_data.loc[self.not_None_rows][col] -
                               self.inferred_data.loc[self.not_None_rows]['lower '+col],
                               self.inferred_data.loc[self.not_None_rows]['upper '+col] -
                               self.inferred_data.loc[self.not_None_rows][col]],
                                                            linestyle='', fmt=".",color= point_color,
                                                            ecolor= ecolor, elinewidth=elinewidth)
            # ax.set_aspect('equal',adjustable="box")
        minimum = np.min((ax.get_xlim(),ax.get_ylim()))
        maximum = np.max((ax.get_xlim(),ax.get_ylim()))
        ax.set_ylabel("Inferred ")
        ax.set_xlabel("True "+ self.columns[column_index])
        #scientific number
        ax.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
        ax.ticklabel_format(style='sci',scilimits=(-2,2),axis='x')
        ax.plot([minimum, maximum], [minimum, maximum], ls="--",  color= line_color)
        if coverage:
            truth = self.truth_data.loc[self.not_None_rows][self.columns[column_index]].tolist()
            num_data = len(truth)
            print("total_number of datasets: ", num_data)
            lower25 = self.inferred_data.loc[self.not_None_rows]['lower25 '+col].tolist()
            upper75 = self.inferred_data.loc[self.not_None_rows]['upper75 '+col].tolist()
            assert len(truth) == len(lower25)
            assert len(upper75) == len(truth)
            count = 0
            average_length =[] # average length of 50% intervals
            for item in range(len(truth)):
                average_length.append(upper75[item]-lower25[item])
                if (lower25[item] <= truth[item]) and \
                        (truth[item] <= upper75[item]):
                    count +=1
            if (count/num_data) >= 0.5:
                pval = min(1, 2*(1- stats.binom(n= num_data, p =0.5).cdf(x=count-1)))
            else:
                pval = 2*(stats.binom(n= num_data, p =0.5).cdf(x=count))
            print("50 percent coverage for ", col," is ", count/num_data, "with pvalue", pval)
            print("The average length for 50 percent interval is ", np.mean(average_length))

    def multi_scatter(self, CI = False, argweaver = False, coverage=True):
        '''
        :param CI: if True--> plot 95 percent interval
        :param argweaver: if True, plot the argweaver output otherwise ARGinfer
        both_inferred: if true draw ARGinfer versus ARGweaver
        :param coverage: provide the coverage of 50% interval
        :return:
        '''
        for ind in range(len(self.columns)):
            self.single_scatter(column_index=ind, CI= CI,
                                argweaver= argweaver, coverage = coverage)
        if not argweaver:
            self.fig.suptitle("ARGinfer")
            figure_name= "scatter"+"ARGinferr"
        else:
            self.fig.suptitle("ARGweaver")
            figure_name= "scatter"+"ARGweaver2"
        if CI:
            figure_name = figure_name+"CI"
        plt.savefig(self.inferred_path+"/{}.pdf".format(figure_name),
                    bbox_inches='tight', dpi=400)
        plt.close()

    def single_std(self, column_index):
        self.gs.update(wspace=0.3, hspace=0.7) # set the spacing between axes.
        if column_index <2:
            ax = self.fig.add_subplot(self.gs[0, column_index])
        elif column_index<4:
            ax = self.fig.add_subplot(self.gs[1, column_index-2])
        elif column_index <6:
            ax = self.fig.add_subplot(self.gs[2, column_index-4])
        else:
            ax = self.fig.add_subplot(self.gs[3, column_index-6])
        if self.columns[column_index] == "ancestral recomb":
            col= 'total recomb'
        else:
            col = self.columns[column_index]
        ax.plot(range(self.inferred_data.loc[self.not_None_rows].shape[0]),
                      self.inferred_data.loc[self.not_None_rows]["std "+self.columns[column_index]],
                      color= "black", label ="ARGinfer")
        ax.plot(range(self.inferred2_data.loc[self.not_None_rows].shape[0]), # inferred 2 is argweaver
                      self.inferred2_data.loc[self.not_None_rows]["std "+col],
                      color= "red", label ="ARGweaver")
        ax.set_ylabel( self.columns[column_index])
        ax.set_xlabel('Data sets')
        #scientific number
        ax.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
        ax.ticklabel_format(style='sci',scilimits=(-2,2),axis='x')
        ax.legend( labelspacing=0,loc='upper right',frameon=False)

    def multi_std(self):
        for ind in range(len(self.columns)):
            self.single_std(column_index=ind)
        self.fig.suptitle("standard deviation of MCMC samples")
        figure_name= "std"

        plt.savefig(self.inferred_path+"/{}.pdf".format(figure_name),
                    bbox_inches='tight', dpi=400)
        plt.close()

def plot_tmrca(truth_path ='', argweaver_path='', arginfer_path='',
               inferred_filename='tmrca.h5'):
    '''a 1*2 plot for tmrca of arginfer and argweaver for same data set'''
    true_tmrca= np.load(truth_path)
    argweaver_tmrca= pd.read_hdf(argweaver_path+"/"+inferred_filename, mode="r")
    arginfer_tmrca= pd.read_hdf(arginfer_path+"/"+inferred_filename, mode="r")
    fig = plt.figure(tight_layout=False)
    gs = gridspec.GridSpec(1, 2)
    #arginfer
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(range(true_tmrca.size),
                  true_tmrca,
                  color= "black", label ="True")
    ax1.plot(range(arginfer_tmrca.shape[0]),
                  arginfer_tmrca["tmrca"],
                  color= "red", label ="Inferred")
    ax1.fill_between(range(arginfer_tmrca.shape[0]),
                    arginfer_tmrca["lower tmrca"],
                    arginfer_tmrca["upper tmrca"],
                    color='lightcoral', alpha=.2)
    # ax1.legend(labelspacing=0,loc='upper right',frameon=False)
    ax1.set_ylabel("TMRCA")
    ax1.set_xlabel('Genomic Position')
    ax1.set_title("ARGinfer")
    #scientific number
    ax1.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
    ax1.ticklabel_format(style='sci',scilimits=(0,0),axis='x')
    #argweaver
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    ax2.plot(range(true_tmrca.size),
                  true_tmrca,
                  color= "black")
    ax2.plot(range(argweaver_tmrca.shape[0]),
                  argweaver_tmrca["tmrca"],
                  color= "red")
    ax2.fill_between(range(argweaver_tmrca.shape[0]),
                    argweaver_tmrca["lower tmrca"],
                    argweaver_tmrca["upper tmrca"],
                    color='lightcoral', alpha=.2)

    # ax2.legend( labelspacing=0,loc='upper right',frameon=False)
    ax2.set_title("ARGweaver")
    ax2.set_xlabel('Genomic Position')
    #scientific number
    ax2.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
    ax2.ticklabel_format(style='sci',scilimits=(0,0),axis='x')
    fig.legend(loc = 'best')#upper right
    figure_name= "tmrca_combined"
    plt.savefig(arginfer_path+"/{}.pdf".format(figure_name),
                bbox_inches='tight', dpi=400)
    plt.close()
    from scipy import stats
    pearson_coef1, p_value1 = stats.pearsonr(true_tmrca, arginfer_tmrca["tmrca"])
    pearson_coef2, p_value2 = stats.pearsonr(true_tmrca, argweaver_tmrca["tmrca"])
    print("arginfer pearson_coef", pearson_coef1, "p_value", p_value1)
    print("argweaver pearson_coef", pearson_coef2, "p_value", p_value2)

# def plot_interval1(argweaver_path=' ', arginfer_path=' ',
#                           columns=["branch length", "ancestral recomb"]):
#     arginfer_data = pd.read_hdf(arginfer_path + '/summary_all.h5', mode="r")
#     weaver_data = pd.read_hdf(argweaver_path + "/summary_all.h5", mode="r")
#     not_None_rows = np.where(arginfer_data['prior'].notnull())[0]
#     arginfer_data = arginfer_data.loc[not_None_rows, : ]
#     weaver_data = weaver_data.loc[not_None_rows, : ]
#     arginfer_data = arginfer_data.reset_index()
#     weaver_data = weaver_data.reset_index()
#     distance= 0.65
#     colors =["darkgreen", "m"]
#     linewidth = 0.75
#     fig = plt.figure(tight_layout=False)
#     gs = gridspec.GridSpec(2, 1)
#     #arginfer
#     ax1 = fig.add_subplot(gs[0, 0])
#     for i in range(min(150,arginfer_data.shape[0])):
#         ax1.vlines(i+1, arginfer_data.loc[i, "lower25 branch length"],
#                    arginfer_data.loc[i, "upper75 branch length"],
#                    linestyles ="solid", colors =colors[0], linewidth= linewidth, label="infer")#dashed
#         ax1.vlines(i+1+distance, weaver_data.loc[i, "lower25 branch length"],
#                    weaver_data.loc[i, "upper75 branch length"],
#                    linestyles ="solid", colors =colors[1], linewidth= linewidth, label = "weaver")
#     # ax1.legend(labelspacing=0,loc='upper right',frameon=False)
#     ax1.set_ylabel("50% CI")
#     ax1.set_xlabel('Data set')
#     ax1.set_title("Total branch length")
#     #scientific number
#     ax1.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
#     ax1.ticklabel_format(style='sci',scilimits=(0,0),axis='x')
#     #argweaver
#     ax2 = fig.add_subplot(gs[1, 0])#, sharey=ax1
#     for i in range(min(arginfer_data.shape[0], 150)):
#         ax2.vlines(i+1, arginfer_data.loc[i, "lower25 ancestral recomb"],
#                    arginfer_data.loc[i, "upper75 ancestral recomb"],
#                    linestyles ="solid", colors =colors[0], linewidth= linewidth, label= "infer")
#         ax2.vlines(i+1+distance, weaver_data.loc[i, "lower25 total recomb"],
#                    weaver_data.loc[i, "upper75 total recomb"],
#                    linestyles ="solid", colors =colors[1], linewidth= linewidth, label = "weaver")
#     # ax1.legend(labelspacing=0,loc='upper right',frameon=False)
#     ax2.set_ylabel("50% CI")
#     ax2.set_xlabel('Data set')
#     ax2.set_title("Ancestral recombination")
#     #scientific number
#     ax2.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
#     ax2.ticklabel_format(style='sci',scilimits=(0,0),axis='x')
#     # ax2.legend( labelspacing=0,loc='upper right',frameon=False)
#     #scientific number
#     # fig.legend(loc = 'best')#upper right
#
#     figure_name= "interval_50_combined"
#     plt.savefig(arginfer_path+"/{}.pdf".format(figure_name),
#                 bbox_inches='tight', dpi=400)
#     plt.close()

def plot_interval(true_path = '', argweaver_path=' ', arginfer_path=' ',
                          columns=["branch length", "ancestral recomb"]):
    true_data = pd.read_hdf(true_path +  "/true_summary.h5", mode="r")
    arginfer_data = pd.read_hdf(arginfer_path + '/summary_all.h5', mode="r")
    weaver_data = pd.read_hdf(argweaver_path + "/summary_all.h5", mode="r")
    not_None_rows = np.where(arginfer_data['prior'].notnull())[0]
    def modify_df(df, keep_rows):
        '''remove rows with NA and reset the indeces'''
        df = df.loc[keep_rows, : ]
        df.reset_index(inplace=True, drop=True)
        return df
    arginfer_data = modify_df(arginfer_data, not_None_rows)
    weaver_data = modify_df(weaver_data, not_None_rows)
    true_data = modify_df(true_data, not_None_rows)
    colors =["black", "red","blue"]
    labels = ["True","ARGinfer", "ARGweaver"]
    num_data=150
    linewidth = 0.6
    alpha = 0.5
    fig = plt.figure(tight_layout=False)
    gs = gridspec.GridSpec(2, 1)
    # plt.title("R = 1")
    #arginfer
    #------ lets first sort data ascending
    true_data.sort_values(by=["branch length"], ascending=True, inplace=True)
    new_index= true_data.index.tolist()
    true_data.reset_index(inplace=True, drop=True)
    #---- reindex the other two based on this
    arginfer_data = arginfer_data.reindex(new_index)
    arginfer_data.reset_index(inplace=True, drop=True)
    weaver_data = weaver_data.reindex(new_index)
    weaver_data.reset_index(inplace=True, drop=True)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(range(num_data),#arginfer_data.shape[0]
                  true_data.loc[range(num_data),"branch length"],
                  color= colors[0], label = labels[0], linewidth=linewidth)
    ax1.fill_between(range(num_data),
                     weaver_data.loc[range(num_data),"lower25 branch length"],
                     weaver_data.loc[range(num_data),"upper75 branch length"],
                     color=colors[2], alpha=alpha, label = labels[2])
    ax1.fill_between(range(num_data),
                     arginfer_data.loc[range(num_data), "lower25 branch length"],
                     arginfer_data.loc[range(num_data),"upper75 branch length"],
                     color=colors[1], alpha=alpha, label = labels[1])

    ax1.set_ylabel("branch length 50% CI")
    # ax1.set_xlabel('data sets')
    ax1.set_title("R = 1")
    # plt.title("R = 1")
    #scientific number
    ax1.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
    # ax1.ticklabel_format(style='sci',scilimits=(0,0),axis='x')
    #argweaver
    #------ lets first sort data ascending
    true_data.sort_values(by=["ancestral recomb"], ascending=True, inplace=True)
    new_index= true_data.index.tolist()
    true_data.reset_index(inplace=True, drop=True)
    #---- reindex the other two based on this
    arginfer_data = arginfer_data.reindex(new_index)
    arginfer_data.reset_index(inplace=True, drop=True)
    weaver_data = weaver_data.reindex(new_index)
    weaver_data.reset_index(inplace=True, drop=True)
    ax2 = fig.add_subplot(gs[1, 0])#, sharey=ax1
    ax2.plot(range(num_data),
                  true_data.loc[range(num_data), "ancestral recomb"],
                  color= colors[0],  linewidth=linewidth)
    ax2.fill_between(range(num_data),
                     weaver_data.loc[range(num_data),"lower25 total recomb"],
                     weaver_data.loc[range(num_data),"upper75 total recomb"],
                     color=colors[2], alpha=alpha)
    ax2.fill_between(range(num_data),
                     arginfer_data.loc[range(num_data),"lower25 ancestral recomb"],
                     arginfer_data.loc[range(num_data),"upper75 ancestral recomb"],
                     color=colors[1], alpha=alpha)#lightcoral

    # ax1.legend(labelspacing=0,loc='upper right',frameon=False)
    ax2.set_ylabel("anc recomb 50% CI")
    ax2.set_xlabel('data sets')
    # ax2.set_title("Ancestral recombination")
    #scientific number
    # ax2.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
    # ax2.ticklabel_format(style='sci',scilimits=(0,0),axis='x')
    # ax2.legend( labelspacing=0,loc='upper right',frameon=False)
    #scientific number
    fig.legend(loc = 'upper right', ncol=3)#best#
    figure_name= "interval_50_combined"
    plt.savefig(arginfer_path+"/{}.pdf".format(figure_name),
                bbox_inches='tight', dpi=400)
    plt.close()

def recrate_boxplot(true_general_path = '/data/projects/punim0594/Ali/phd/mcmc_out/sim10L100K',
                    argweaver_general_path=' ',
                    arginfer_general_path=' '):

    trueR1_path= true_general_path+"/sim_r1"
    trueR2_path= true_general_path+"/sim_r2"
    trueR4_path= true_general_path+"/sim_r4"
    weaverR1_path = argweaver_general_path+"/r1/n10L100K"
    weaverR2_path = argweaver_general_path+"/r2/n10L100K"
    weaverR4_path = argweaver_general_path+"/r4/n10L100K"
    inferR1_path = arginfer_general_path+ "/n10L100K_r1"
    inferR2_path = arginfer_general_path+ "/n10L100K_r2"
    inferR4_path = arginfer_general_path+ "/n10L100K_r4"
    def modify_df(infer_df, weaver_df, keep_rows, R =1):
        '''remove rows with NA and reset the indeces'''
        infer_df = infer_df[["branch length", "ancestral recomb"]]
        weaver_df = weaver_df[["branch length", "total recomb"]]
        infer_df = infer_df.loc[keep_rows, : ]
        weaver_df = weaver_df.loc[keep_rows, : ]
        weaver_df["ancestral recomb"] = weaver_df["total recomb"]
        del weaver_df["total recomb"]
        weaver_df["model"] ="ARGweaver"
        infer_df["model"] = "ARGinfer"
        both_df = pd.concat((infer_df, weaver_df))
        both_df["R"] = R
        both_df["recombination rate"] = both_df["ancestral recomb"]/both_df["branch length"]
        return both_df
    #R1 data
    # "/true_summary.h5", mode="r"

    inferR1_data = pd.read_hdf(inferR1_path + '/summary_all.h5', mode="r")
    weaverR1_data = pd.read_hdf(weaverR1_path + "/summary_all.h5", mode="r")
    not_None_rowsR1 = np.where(inferR1_data['prior'].notnull())[0]
    R1_data = modify_df(inferR1_data,weaverR1_data, not_None_rowsR1, R=1)
    true_R1= pd.read_hdf(trueR1_path + '/true_summary.h5', mode="r")
    true_R1= true_R1.loc[not_None_rowsR1, : ]

    # R2 data
    inferR2_data = pd.read_hdf(inferR2_path + '/summary_all.h5', mode="r")
    weaverR2_data = pd.read_hdf(weaverR2_path + "/summary_all.h5", mode="r")
    not_None_rowsR2 = np.where(inferR2_data['prior'].notnull())[0]
    R2_data = modify_df(inferR2_data,weaverR2_data, not_None_rowsR2, R=2)
    true_R2= pd.read_hdf(trueR2_path + '/true_summary.h5', mode="r")
    true_R2= true_R2.loc[not_None_rowsR2, : ]
    #-- R4
    inferR4_data = pd.read_hdf(inferR4_path + '/summary_all.h5', mode="r")
    weaverR4_data = pd.read_hdf(weaverR4_path + "/summary_all.h5", mode="r")
    not_None_rowsR4 = np.where(inferR4_data['prior'].notnull())[0]
    R4_data = modify_df(inferR4_data,weaverR4_data, not_None_rowsR4, R=4)
    true_R4= pd.read_hdf(trueR4_path + '/true_summary.h5', mode="r")
    true_R4= true_R4.loc[not_None_rowsR4, : ]

    #------combine all together
    data_df= pd.concat((R1_data, R2_data,R4_data))
    sns.boxplot(x='R', y='recombination rate', hue='model', data=data_df)
    plt.axhline(y= np.mean(true_R1["ancestral recomb"]/true_R1["branch length"]),
                linewidth=1, color='r', linestyle = "--", xmin=0, xmax=.5)#1e-8
    plt.axhline(y=np.mean(true_R2["ancestral recomb"]/true_R2["branch length"]),
                linewidth=1, color='r', linestyle = "--", xmin=0, xmax=.8)#0.5e-8
    plt.axhline(y=np.mean(true_R4["ancestral recomb"]/true_R4["branch length"]),
                linewidth=1, color='r', linestyle = "--", xmin=0, xmax=1)#0.25e-8
    plt.legend(loc='upper right')
    figure_name= "recombRateBoxplot"
    plt.title("Box plot of recombination rate")
    plt.savefig(inferR1_path+"/{}.pdf".format(figure_name),
                bbox_inches='tight', dpi=400)


if __name__=='__main__':
    # s= Scatter(truth_path= '/data/projects/punim0594/Ali/phd/mcmc_out/sim10L100K/sim_r4',
    #            inferred_path='/data/projects/punim0594/Ali/phd/mcmc_out/ARGinfer/M2/n10L100K_r4',
    #            columns=["branch length", 'total recomb', "ancestral recomb", 'posterior'])
    # s.multi_scatter(CI=True, argweaver= False, coverage = True)
    # s= Scatter(truth_path = '/data/projects/punim0594/Ali/phd/mcmc_out/sim10L100K/sim_r4',
    #            inferred_path='/data/projects/punim0594/Ali/phd/mcmc_out/aw/r4/n10L100K',
    #            columns=["branch length", "ancestral recomb"])
    # s.multi_scatter(CI=True, argweaver= True, coverage = True)
    # s= Scatter(truth_path= '/data/projects/punim0594/Ali/phd/mcmc_out/aw/r2/n10L100K',
    #            inferred_path='/data/projects/punim0594/Ali/phd/mcmc_out/ARGinfer/n10L100K_r2',
    #            columns=["branch length", 'total recomb', "ancestral recomb", 'posterior'], std=True)
    # s.multi_std()
    plot_tmrca(truth_path ='/data/projects/punim0594/Ali/phd/mcmc_out/sim10L100K/sim_r2/true_tmrca3.npy',
                   arginfer_path='/data/projects/punim0594/Ali/phd/mcmc_out/ARGinfer/M2/n10L100K_r2/out3',
                    argweaver_path = '/data/projects/punim0594/Ali/phd/mcmc_out/aw/r2/n10L100K/out3',
                   inferred_filename='tmrca.h5')

    # plot_interval(true_path= '/data/projects/punim0594/Ali/phd/mcmc_out/sim10L100K/sim_r1',
    #                 argweaver_path='/data/projects/punim0594/Ali/phd/mcmc_out/aw/r1/n10L100K',
    #               arginfer_path='/data/projects/punim0594/Ali/phd/mcmc_out/ARGinfer/M2/n10L100K_r1',
    #                       columns=["branch length", "ancestral recomb"])

    # recrate_boxplot(argweaver_general_path='/data/projects/punim0594/Ali/phd/mcmc_out/aw',
    #                 arginfer_general_path='/data/projects/punim0594/Ali/phd/mcmc_out/ARGinfer')
