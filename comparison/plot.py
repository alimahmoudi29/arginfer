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
from tqdm import tqdm

def coverage50_and_average_length_mse(truth ,inferred_mean,
                                      inferred_lower25,inferred_upper75):
    '''
    :param truth: an np.array or a pd df column of the true values
    :param inferred_mean: a pd df or a np array of the mean of the inferred
    :param inferred_lower25: an np array or a pd df of quantile 0.25
    :param inferred_upper75: an no array or a pd df of quantile 0.75
    :return: 50% Coverage probability, 50%average length, mean square error
    '''
    coverage50 = sum((inferred_lower25 <=truth)& \
                   (truth<= inferred_upper75))/len(truth)
    average_length = np.mean(inferred_upper75 - inferred_lower25)
    mse=  sum((inferred_mean- truth)**2)/len(truth)

    return coverage50, average_length, mse

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

    def __init__(self, truth_path, inferred_path, inferred2_path='',
                 columns = ["likelihood","prior", "posterior", "branch length"],
                 std = False, weaver_infer=False):
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
        # print("plot size in inch:",plt.rcParams.get('figure.figsize'))
        self.gs = gridspec.GridSpec(int(plot_dimension[0]), int(plot_dimension[1]))
        if not std:
            self.truth_data = pd.read_hdf(self.truth_path +  "/true_summary.h5", mode="r")
            self.inferred_data = pd.read_hdf(self.inferred_path + '/summary_all.h5', mode="r")
        else: # both inferred, note that instead of truth
            self.inferred_data = pd.read_hdf(self.inferred_path + '/summary_all.h5', mode="r")
            self.inferred2_data = pd.read_hdf(self.truth_path + "/summary_all.h5", mode="r")
        # ------
        if weaver_infer:
            self.inferred2_data = pd.read_hdf(inferred2_path + "/summary_all.h5", mode="r")
            self.fig = plt.figure(tight_layout=False, figsize=(7,3))
            print("plot size in inch:",plt.rcParams.get('figure.figsize'))
            self.gs = gridspec.GridSpec(1, 2)
        #row indexes with not None values
        self.not_None_rows = np.where(self.inferred_data['prior'].notnull())[0]


    def ARGinfer_weaver(self,  R, both_infer_methods=False):
        '''for branch length and ancestral recombination
        plot arginfer and argweaver in a (1, 2) plot
        each of them are compared against the truth
        NOTE: make sure, argument weaver_infer=True.
        '''
        self.gs.update(wspace=0, hspace=0.4)
        if self.columns[0] =="branch length":
            weaver_col= "branch length"
            self.fig.suptitle("Total branch length, "+ r"$\theta/\rho = $"+ str(R), fontsize=9)
        if self.columns[0] =="ancestral recomb":
            weaver_col = "total recomb"
            self.fig.suptitle("Ancestral recombination, "+ r"$\theta/\rho = $"+ str(R), fontsize=9)
        truth_data = self.truth_data.loc[self.not_None_rows]
        infer_data= self.inferred_data.loc[self.not_None_rows]
        weaver_data= self.inferred2_data.loc[self.not_None_rows]
        line_color= "black"
        point_color= ["#483d8b","#a52a2a"]
        ecolor = ["#6495ed","#ff7f50"]##dc143c #8a2be2#00ffff
        elinewidth = 0.55
        markersize='4'
        linewidth= 0.2
        fontsize=10
        if not both_infer_methods: # then versus the truth
            ax1 = self.fig.add_subplot(self.gs[0, 0])
            ax1.errorbar(truth_data[self.columns[0]],
                        infer_data[self.columns[0]],
                        yerr= [infer_data[self.columns[0]] -
                               infer_data['lower '+self.columns[0]],
                               infer_data['upper '+self.columns[0]] -
                               infer_data[self.columns[0]]],
                               linestyle='', fmt=".",color= point_color[0],
                               ecolor= ecolor[0], elinewidth=elinewidth,
                         markersize = markersize, linewidth= linewidth)
                # ax.set_aspect('equal',adjustable="box")
            minimum = np.min((ax1.get_xlim(),ax1.get_ylim()))
            maximum = np.max((ax1.get_xlim(),ax1.get_ylim()))
            ax1.set_title("ARGinfer", fontsize= fontsize)
            ax1.set_ylabel("Inferred ",  fontsize= fontsize)
            ax1.set_xlabel("True "+ self.columns[0],  fontsize= fontsize)
            # remove values in axis x
            ax1.set_xticklabels([])
            # ax1.xaxis.set_visible(False)
            #scientific number
            ax1.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
            # ax1.ticklabel_format(style='sci',scilimits=(-2,2),axis='x')
            ax1.plot([minimum, maximum], [minimum, maximum], ls="--",  color= line_color, label ="Truth")
            self.gs.update(wspace=0.2, hspace=0.7)
            ax2 = self.fig.add_subplot(self.gs[0, 1])
            ax2.errorbar(truth_data[self.columns[0]],
                        weaver_data[weaver_col],
                        yerr= [weaver_data[weaver_col] -
                               weaver_data['lower '+weaver_col],
                               weaver_data['upper '+weaver_col] -
                               weaver_data[weaver_col]],
                               linestyle='', fmt=".",color= point_color[1],
                                ecolor= ecolor[1], elinewidth=elinewidth,
                         markersize= markersize, linewidth= linewidth)
                # ax.set_aspect('equal',adjustable="box")
            minimum = np.min((ax2.get_xlim(),ax2.get_ylim()))
            maximum = np.max((ax2.get_xlim(),ax2.get_ylim()))
            self.fig.subplots_adjust(top=0.78)
            ax2.set_title("ARGweaver", fontsize= fontsize)
            ax2.set_ylabel("Inferred ", fontsize= fontsize)
            ax2.set_xlabel("True "+ self.columns[0], fontsize= fontsize)
            # remove the axis values
            ax2.set_yticklabels([])
            ax2.set_xticklabels([])
            # ax2.axis("off")
            # ax2.yaxis.set_visible(False)
            #scientific number
            # ax2.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
            # ax2.ticklabel_format(style='sci',scilimits=(-2,2),axis='x')
            ax2.plot([minimum, maximum], [minimum, maximum], ls="--",  color= line_color, label ="Truth")
            figure_name= "BothScatter"+self.columns[0]+ str(R)
        else: # ARGweaver versus ARGinfer
            line_color= "red"
            markersize='6'
            linewidth= 0.1
            self.fig.suptitle(r"$\theta/\rho = $"+ str(R), fontsize=9)
            self.gs.update(wspace=0.3, hspace=0.4)
            ax1 = self.fig.add_subplot(self.gs[0, 0])
            ax1.errorbar(infer_data["branch length"],
                        weaver_data["branch length"],
                               linestyle='', fmt=".",color= point_color[0],
                         markersize = markersize, linewidth= linewidth)
                # ax.set_aspect('equal',adjustable="box")
            minimum = np.min((ax1.get_xlim(),ax1.get_ylim()))
            maximum = np.max((ax1.get_xlim(),ax1.get_ylim()))
            ax1.set_title("Total branch length", fontsize= fontsize)
            ax1.set_ylabel("ARGweaver ",  fontsize= fontsize)
            ax1.set_xlabel("ARGinfer" ,  fontsize= fontsize)
            # remove values in axis x
            # ax1.set_xticklabels([])
            # ax1.xaxis.set_visible(False)
            #scientific number
            ax1.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
            ax1.ticklabel_format(style='sci',scilimits=(0,0),axis='x')
            ax1.plot([minimum, maximum], [minimum, maximum], ls="--",  color= line_color, label ="Truth")
            self.gs.update(wspace=0.3, hspace=0.7)
            ax2 = self.fig.add_subplot(self.gs[0, 1])
            ax2.errorbar(infer_data["ancestral recomb"],
                        weaver_data["total recomb"],
                         linestyle='', fmt=".",color= point_color[0],
                         markersize= markersize, linewidth= linewidth)
                # ax.set_aspect('equal',adjustable="box")
            minimum = np.min((ax2.get_xlim(),ax2.get_ylim()))
            maximum = np.max((ax2.get_xlim(),ax2.get_ylim()))
            self.fig.subplots_adjust(top=0.88)
            ax2.set_title("Ancestral recombination", fontsize= fontsize)
            ax2.set_ylabel("ARGweaver ", fontsize= fontsize)
            ax2.set_xlabel("ARGinfer", fontsize= fontsize)
            # remove the axis values
            # ax2.set_yticklabels([])
            # ax2.set_xticklabels([])
            # ax2.axis("off")
            # ax2.yaxis.set_visible(False)
            #scientific number
            ax2.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
            ax2.ticklabel_format(style='sci',scilimits=(-1,1),axis='x')
            ax2.plot([minimum, maximum], [minimum, maximum], ls="--",  color= line_color, label ="Truth")
            figure_name= "bothinferenceModels"+ str(R)

        plt.savefig(self.inferred_path+"/{}.pdf".format(figure_name),
                    bbox_inches='tight', dpi=400)
        plt.close()


    def single_scatter(self, column_index, CI = False,
                       argweaver= False, coverage =True):
        line_color= "red"
        point_color= "black"
        ecolor = "purple"
        elinewidth = 0.9
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
            coverage50, average_length, mse = coverage50_and_average_length_mse(truth ,
                                      self.inferred_data.loc[self.not_None_rows][col],
                                      self.inferred_data.loc[self.not_None_rows]['lower25 '+col],
                                      self.inferred_data.loc[self.not_None_rows]['upper75 '+col])
            print("stats for ", col)
            print("coverage50: ", coverage50, "average_length: ",average_length, "mse: ",mse)


    def multi_scatter(self, CI = False, argweaver = False, coverage=True, R=1):
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
            self.fig.suptitle("ARGinfer, "+ r"$\theta\/rho = $"+ str(R))
            figure_name= "scatter"+"ARGinferr"+str(R)
        else:
            self.fig.suptitle("ARGweaver, "+ r"$\theta\/rho = $"+ str(R))
            figure_name= "scatter"+"ARGweaver" +str(R)
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
               inferred_filename='tmrca.h5', CI=95, R=1):
    '''a 1*2 plot for tmrca of arginfer and argweaver for same data set'''
    true_tmrca= np.load(truth_path)
    argweaver_tmrca= pd.read_hdf(argweaver_path+"/"+inferred_filename, mode="r")
    arginfer_tmrca= pd.read_hdf(arginfer_path+"/"+inferred_filename, mode="r")
    fig = plt.figure(tight_layout=False, figsize=(7, 2))
    fig.suptitle(r"$\theta/\rho = $"+ str(R), fontsize=9)
    fig.subplots_adjust(top=0.8)
    gs = gridspec.GridSpec(1, 2)
    #arginfer
    linewidth=.9
    fontsize=8
    alpha=0.3
    bbox_to_anchor=(.85, 1.15)
    # bbox_to_anchor=(.84, 1.22)
    if CI==50:
        lcol="lower25 tmrca"
        ucol="upper75 tmrca"
        fill_label= "50% CI"
    else:#95 percent
        lcol="lower tmrca"
        ucol="upper tmrca"
        fill_label= "95% CI"
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(range(true_tmrca.size),
                  true_tmrca,
                  color= "black", label ="True", linewidth=linewidth,
                    linestyle="dashed")
    ax1.plot(range(arginfer_tmrca.shape[0]),
                  arginfer_tmrca["tmrca"],
                  color= "red", label ="Inferred", linewidth=linewidth)
    ax1.fill_between(range(arginfer_tmrca.shape[0]),
                    arginfer_tmrca[lcol],
                    arginfer_tmrca[ucol],
                    color='lightcoral', alpha=alpha, label = fill_label)
    # ax1.legend(labelspacing=0,loc='upper right',frameon=False)
    ax1.set_ylabel("TMRCA", fontsize=fontsize)
    ax1.set_xlabel('Genomic Position', fontsize=fontsize)
    ax1.set_title("ARGinfer", fontsize=fontsize)
    ax1.tick_params(axis="both", labelsize=7)
    #scientific number
    # ax1.set_yscale('symlog')
    # ax1.set_xscale('symlog')
    ax1.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
    ax1.ticklabel_format(style='sci',scilimits=(0,0),axis='x')
    #argweaver
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    ax2.plot(range(true_tmrca.size),
                  true_tmrca,
                  color= "black", linewidth=linewidth, linestyle="dashed")
    ax2.plot(range(argweaver_tmrca.shape[0]),
                  argweaver_tmrca["tmrca"],
                  color= "red", linewidth=linewidth)
    ax2.fill_between(range(argweaver_tmrca.shape[0]),
                    argweaver_tmrca[lcol],
                    argweaver_tmrca[ucol],
                    color='lightcoral', alpha=alpha)

    # ax2.legend( labelspacing=0,loc='upper right',frameon=False)
    ax2.set_title("ARGweaver",fontsize=fontsize)
    ax2.set_xlabel('Genomic Position',fontsize=fontsize)
    #scientific number
    # ax2.set_yscale('symlog')
    # ax2.set_xscale('symlog')
    ax2.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
    ax2.ticklabel_format(style='sci',scilimits=(0,0),axis='x')
    # fig.legend(loc = 'best')#upper right
    fig.legend(loc = 'upper right', bbox_to_anchor=bbox_to_anchor,
               fancybox=True,fontsize=6)
    ax2.tick_params(axis="both", labelsize=7)
    figure_name= "tmrca_combined"+str(CI)
    plt.savefig(arginfer_path+"/{}.pdf".format(figure_name),
                bbox_inches='tight', dpi=400)
    plt.close()
    from scipy import stats
    pearson_coef1, p_value1 = stats.pearsonr(true_tmrca, arginfer_tmrca["tmrca"])
    pearson_coef2, p_value2 = stats.pearsonr(true_tmrca, argweaver_tmrca["tmrca"])
    infer_coverage50, inferAverage_length, infer_MSE = coverage50_and_average_length_mse(true_tmrca ,
                                                       arginfer_tmrca["tmrca"],
                                                        arginfer_tmrca["lower25 tmrca"],
                                                        arginfer_tmrca["upper75 tmrca"])
    weaver_coverage50, weaverAverage_length, weaver_MSE = coverage50_and_average_length_mse(true_tmrca,
                                                       argweaver_tmrca["tmrca"],
                                                        argweaver_tmrca["lower25 tmrca"],
                                                        argweaver_tmrca["upper75 tmrca"])
    print("ARGinfer Statistics:\n")
    print("Pearson Corr: ", pearson_coef1, "p_value: ", p_value1)
    print("coverage50: ", infer_coverage50, "Avg_len: ", inferAverage_length, "MSE: ", infer_MSE)
    print("-"*20)
    print("ARGweaver Statistics:\n")
    print("Pearson Corr: ", pearson_coef2, "p_value: ", p_value2)
    print("coverage50: ", weaver_coverage50, "Avg_len: ", weaverAverage_length, "MSE: ", weaver_MSE)

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
                          columns=["branch length", "ancestral recomb"], R=1):
    '''plot the 50%CI for all the data sets:
    all ARGinfer, ARGweaver and truth side by side
    '''
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
    colors=["#0000ff","#008b8b", "#ff1493"]#
    # colors =["black", "red","blue"]
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
    ax1.set_title(r"$\theta/\rho = $"+ str(R))
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
    # sns.set_theme(style="dark")
    # sns.set_context("talk")
    #colors: https://hashtagcolor.com
    flatui = ["#228b22", "#4169e1"]##00ff7f
    sns.set_palette(sns.color_palette(flatui))
    sns.boxplot(x='R', y='recombination rate', hue='model', data=data_df,
                linewidth=0.75, showfliers=False, width=0.5)
    plt.axhline(y= np.mean(true_R1["ancestral recomb"]/true_R1["branch length"]),
                linewidth=0.75, color='r', linestyle = "--", xmin=0, xmax=.5,  label="Truth")#1e-8
    plt.axhline(y=np.mean(true_R2["ancestral recomb"]/true_R2["branch length"]),
                linewidth=0.75, color='r', linestyle = "--", xmin=0, xmax=.8)#0.5e-8
    plt.axhline(y=np.mean(true_R4["ancestral recomb"]/true_R4["branch length"]),
                linewidth=0.75, color='r', linestyle = "--", xmin=0, xmax=1)#0.25e-8
    plt.legend(loc='upper right')
    plt.xlabel(r"$\theta/ \rho$")#, size=10
    plt.ylabel(" Recombination rate")
    figure_name= "recombRateBoxplot"
    plt.title("recombination rate estimation")
    plt.savefig(inferR1_path+"/{}.pdf".format(figure_name),
                bbox_inches='tight', dpi=400)

def plot_allele_age(truth_path ='', argweaver_path='', arginfer_path='',
               inferred_filename='allele_age.h5', R=1, CI= 50):
    true_alAge= pd.read_hdf(truth_path, mode="r")
    argweaver_alAge= pd.read_hdf(argweaver_path+"/"+inferred_filename, mode="r")
    arginfer_alAge= pd.read_hdf(arginfer_path+"/"+inferred_filename, mode="r")
    #sort ascending
    true_alAge.sort_values(by=["mid age"], ascending=True, inplace=True)
    new_index= true_alAge.index.tolist()
    true_alAge.reset_index(inplace=True, drop=True)
    #---- reindex the other two based on this
    arginfer_alAge = arginfer_alAge.reindex(new_index)
    arginfer_alAge.reset_index(inplace=True, drop=True)
    argweaver_alAge = argweaver_alAge.reindex(new_index)
    argweaver_alAge.reset_index(inplace=True, drop=True)

    fig = plt.figure(tight_layout=False, figsize=(7, 2))
    gs = gridspec.GridSpec(1, 2)
    #arginfer
    linewidth=.9
    fontsize=9
    if CI==50:
        infer_lcol="lower25 age"
        weaver_lcol= infer_lcol
        infer_ucol="upper75 age"
        weaver_ucol= infer_ucol
        bbox_to_anchor=(0.09, .97)
        fill_label= "50% CI"
    else:#95 percent
        infer_lcol="lower025 age"
        weaver_lcol= "lower age"
        infer_ucol="upper975 age"
        weaver_ucol= "upper age"
        bbox_to_anchor=(0.07, .97)
        fill_label= "95% CI"
    # plt.title('Allele age '+ r"$\theta/\rho$ ="+str(R),fontsize=9)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(range(true_alAge.shape[0]),
                  true_alAge["mid age"],
                  color= "#dc143c", label ="True", linewidth= linewidth, linestyle="dashed")
    ax1.plot(range(arginfer_alAge.shape[0]),
                  arginfer_alAge["mid age"],
                  color= "#0000cd", label ="Inferred", linewidth= linewidth)
    ax1.fill_between(range(arginfer_alAge.shape[0]),
                    arginfer_alAge[infer_lcol],#lower025
                    arginfer_alAge[infer_ucol],#upper975
                    color='#1e90ff', alpha=.5, label = fill_label )
    # ax1.legend(labelspacing=0,loc='upper right',frameon=False)
    ax1.set_ylabel("Allele Age", fontsize=fontsize)
    ax1.set_xlabel('SNP', fontsize=fontsize)
    ax1.set_title("ARGinfer",fontsize=fontsize)
    #scientific number
    ax1.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
    ax1.ticklabel_format(style='sci',scilimits=(0,0),axis='x')
    #argweaver
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    ax2.plot(range(true_alAge.shape[0]),
                  true_alAge["mid age"],
                  color= "#dc143c", linewidth= linewidth, linestyle="dashed")
    ax2.plot(range(argweaver_alAge.shape[0]),
                  argweaver_alAge["mid age"],
                  color= "#0000cd", linewidth=linewidth)
    ax2.fill_between(range(argweaver_alAge.shape[0]),
                    argweaver_alAge[weaver_lcol],
                    argweaver_alAge[weaver_ucol],
                    color='#1e90ff', alpha=.5)

    # ax2.legend( labelspacing=0,loc='upper right',frameon=False)
    ax2.set_title("ARGweaver", fontsize=fontsize)
    ax2.set_xlabel('SNP', fontsize=fontsize)
    #scientific number
    ax2.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
    ax2.ticklabel_format(style='sci',scilimits=(0,0),axis='x')
    # fig.legend(loc = 'best')#upper right
    fig.legend(loc = 'center left', bbox_to_anchor=bbox_to_anchor,
               fancybox=True,fontsize=6, ncol=3)
    figure_name= "allele_age_combined"+ str(CI)
    plt.savefig(arginfer_path+"/{}.pdf".format(figure_name),
                bbox_inches='tight', dpi=400)
    plt.close()

    pearson_coef1, p_value1 = stats.pearsonr(true_alAge["mid age"], arginfer_alAge["mid age"])
    pearson_coef2, p_value2 = stats.pearsonr(true_alAge["mid age"], argweaver_alAge["mid age"])
    infer_coverage50, inferAverage_length, infer_MSE = coverage50_and_average_length_mse(true_alAge["mid age"] ,
                                                       arginfer_alAge["mid age"],
                                                        arginfer_alAge["lower25 age"],
                                                        arginfer_alAge["upper75 age"])
    weaver_coverage50, weaverAverage_length, weaver_MSE = coverage50_and_average_length_mse(true_alAge["mid age"],
                                                       argweaver_alAge["mid age"],
                                                        argweaver_alAge["lower25 age"],
                                                        argweaver_alAge["upper75 age"])
    print("ARGinfer Statistics:\n")
    print("Pearson Corr: ", pearson_coef1, "p_value: ", p_value1)
    print("coverage50: ", infer_coverage50, "Avg_len: ", inferAverage_length, "MSE: ", infer_MSE)
    print("-"*20)
    print("ARGweaver Statistics:\n")
    print("Pearson Corr: ", pearson_coef2, "p_value: ", p_value2)
    print("coverage50: ", weaver_coverage50, "Avg_len: ", weaverAverage_length, "MSE: ", weaver_MSE)



def get_summary_tmrca_allele_age(truth_general_path ='',
                                            arginfer_general_path='',
                                            argweaver_general_path ='',
                                            replicate=161,
                                            feature = "tmrca", R=1):
    '''report the  50% coverage prob, 50% average length, MSE, and Pearson
     for tmrca or allele age.
     '''
    coverage_df= pd.DataFrame(columns=["coverage", "average length",
                                       "mse", "pearson", "model"])
    for i in tqdm(range(replicate), ncols=100, ascii=False):
        if os.path.isfile(arginfer_general_path+"/out" +str(i)+"/"+feature+".h5"):
            if feature == "tmrca":
                truth_data= np.load(truth_general_path+"/true_tmrca"+str(i)+".npy")
                col ="tmrca"
                prefix=''
            elif feature =="allele_age":
                truth_data = pd.read_hdf(truth_general_path+"/true_allele_age"
                                         +str(i)+".h5", mode="r")["mid age"]
                col="age"
                prefix ="mid "
            else:
                raise IOError("wrong feature! this is only for tmrca and alele_age")
            infer_data = pd.read_hdf(arginfer_general_path+"/out" +str(i)+
                                     "/"+feature+".h5", mode="r")
            weaver_data = pd.read_hdf(argweaver_general_path+"/out" +str(i)+
                                      "/"+feature+".h5", mode="r")
            # ARGinfer
            pearson_coef1, p_value1 = stats.pearsonr(truth_data, infer_data[prefix+col])
            infer_cover50, inferAvgLen, infer_MSE = coverage50_and_average_length_mse(truth_data ,
                                                               infer_data[prefix+col],
                                                                infer_data["lower25 "+col],
                                                                infer_data["upper75 "+col])
            coverage_df.loc[coverage_df.shape[0]] =[infer_cover50, inferAvgLen, infer_MSE,
                                                    pearson_coef1, "ARGinfer"]
            #ARGweaver
            pearson_coef2, p_value2 = stats.pearsonr(truth_data, weaver_data[prefix+col])
            weaver_cover50, weaverAvgLen, weaver_MSE = coverage50_and_average_length_mse(truth_data,
                                                               weaver_data[prefix+col],
                                                                weaver_data["lower25 "+col],
                                                                weaver_data["upper75 "+col])
            coverage_df.loc[coverage_df.shape[0]] =[weaver_cover50, weaverAvgLen, weaver_MSE,
                                                    pearson_coef2, "ARGweaver"]
        else:
            pass
            # coverage_df.loc[coverage_df.shape[0]] =[None for i in range(5)]
    coverage_df.to_hdf(arginfer_general_path + "/coverage_all"+feature+".h5", key = "df")
    #-------- do some ploting
    infer_df = coverage_df[coverage_df["model"]=="ARGinfer"]
    infer_df.reset_index(inplace=True, drop=True)
    weaver_df = coverage_df[coverage_df["model"]=="ARGweaver"]
    weaver_df.reset_index(inplace=True, drop=True)
    #sort ascending
    def sort_df(infer_df,weaver_df,  column):
        '''sort infer_df by columns ascending and then
         rearrange weaver_df based on that'''
        infer_df.sort_values(by=[column], ascending=True, inplace=True)
        new_index= infer_df.index.tolist()
        infer_df.reset_index(inplace=True, drop=True)
        #---- reindex the other two based on this
        weaver_df = weaver_df.reindex(new_index)
        weaver_df.reset_index(inplace=True, drop=True)
        return infer_df, weaver_df
    infer_df, weaver_df =sort_df(infer_df,weaver_df, column="coverage")
    #------ plot

    # fig = plt.figure(tight_layout=False)
    # gs = gridspec.GridSpec(1, 2)
    # #arginfer
    # ax1 = fig.add_subplot(gs[0, 0])
    # ax1.plot(range(infer_df.shape[0]),
    #               infer_df["coverage"],
    #               color= "red", label ="ARGinfer")
    # ax1.plot(range(weaver_df.shape[0]),
    #               weaver_df["coverage"],
    #               color= "b", label ="ARGweaver")
    # ax2 = fig.add_subplot(gs[0, 1])
    # infer_df, weaver_df =sort_df(infer_df,weaver_df, column="average length")
    # ax2.plot(range(infer_df.shape[0]),
    #               infer_df["average length"],
    #               color= "red")
    # ax2.plot(range(weaver_df.shape[0]),
    #               weaver_df["average length"],
    #               color= "b")

    # fig.legend(loc = 'best')#upper right

    # #arginfer
    # flatui = ["#00ff7f", "#4169e1"]
    # sns.set_palette(sns.color_palette(flatui))
    # fig, ax_new = plt.subplots(1,4, sharey=False)
    # layout = (1,4)
    #
    #
    # bp = coverage_df.boxplot(by="model",ax=ax_new,layout=layout,
    #                          patch_artist = True,figsize=(4,4),  fontsize=7,grid=False )#,figsize=(6,8), vert = 0
    # [ax_tmp.set_xlabel('') for ax_tmp in ax_new.reshape(-1)]
    # [ax_tmp.get_xaxis().set_visible(False) for ax_tmp in ax_new.reshape(-1)]

    # # [ax_tmp.set_xlabel('') for ax_tmp in ax_new.reshape(-1)]
    # # [ax_tmp.set_ylim(-2, 2) for ax_tmp in ax_new[1]]
    # fig.suptitle(r'$\theta/\rho$='+str(R))


    # coverage_df.boxplot(column=['coverage', 'average length', 'mse', "pearson"], by=['model'])
    # ax1 = fig.add_subplot(gs[0, 0])
    # flatui = ["#00ff7f", "#4169e1"]
    # sns.set_palette(sns.color_palette(flatui))
    # # sns.boxplot(x="variable", y="value", hue='model', data=pd.melt(coverage_df),linewidth=0.75)
    # sns.boxplot(x= "model", y='mse', hue='model', data=coverage_df,
    #             linewidth=0.75)
    figure_name= "coverageAll"+str(replicate)+feature
    plt.savefig(arginfer_general_path+"/{}.pdf".format(figure_name),
                bbox_inches='tight', dpi=400)
    plt.close()

    #print some stats
    infer_mean = coverage_df[coverage_df["model"]=="ARGinfer"].mean().tolist()
    weaver_mean = coverage_df[coverage_df["model"]=="ARGweaver"].mean().tolist()
    infer_median = coverage_df[coverage_df["model"]=="ARGinfer"].median().tolist()
    weaver_median = coverage_df[coverage_df["model"]=="ARGweaver"].median().tolist()
    print("feature is:", feature)
    print("ARGinfer Statistics:\n")
    print("Pearson Corr mean: ", infer_mean[3], "median:", infer_median[3])
    print("coverage50 mean: ", infer_mean[0], "median:",infer_median[0], "\nAvg_len mean: ",
          infer_mean[1],infer_median[1], "\nMSE mean: ", infer_mean[2], "median:", infer_median[2])
    print("-"*20)
    print("ARGweaver Statistics:\n")
    print("Pearson Corr mean: ", weaver_mean[3], "median", weaver_median[3])
    print("coverage50 mean: ", weaver_mean[0], "median", weaver_median[0],
          "\nAvg_len mean: ", weaver_mean[1], "median",weaver_median[1],
          "\nMSE: ", weaver_mean[2], "median:", weaver_median[2])

    # now we can plot it

def compare_infer_weaver(truth_path ='', argweaver_path='', arginfer_path='',
                         column = ["branch length"], sub_sample =50):
        '''
        only one scatter plot for  a sub set (sub_sample) of data sets to
        plot both argweaver and arginfer side by side
        This is for the completion seminar
        '''
        truth_data = pd.read_hdf(truth_path +  "/true_summary.h5", mode="r")
        infer_data = pd.read_hdf(arginfer_path + '/summary_all.h5', mode="r")
        weaver_data = pd.read_hdf(argweaver_path + '/summary_all.h5', mode="r")
        fig = plt.figure(tight_layout=False, figsize=(8, 3))
        print("plot size in inch:",plt.rcParams.get('figure.figsize'))
        gs = gridspec.GridSpec(1, 1)
        not_None_rows = np.where(infer_data['prior'].notnull())[0]

        #sub sample:
        truth_data = truth_data.loc[not_None_rows][0:sub_sample]
        infer_data= infer_data.loc[not_None_rows][0:sub_sample]
        weaver_data= weaver_data.loc[not_None_rows][0:sub_sample]
        if column[0] =="branch length":
            weaver_col= "branch length"
            epsilon= 50000000
        if column[0] =="ancestral recomb":
            weaver_col = "total recomb"
            epsilon = 0.3

        # line_color= "black"
        # point_color= ["#483d8b","#a52a2a"]
        # ecolor = ["#6495ed","#ff7f50"]##dc143c #8a2be2#00ffff
        line_color= "black"
        point_color= ["blue", "orange"]#orange
        ecolor = ["blue", "darkorange"]
        markersize='4'
        elinewidth = 0.72
        gs.update(wspace=0.3, hspace=0.7) # set the spacing between axes.
        ax = fig.add_subplot(gs[0, 0])
        ax.errorbar(truth_data[column[0]],
                    infer_data[column[0]],
                    yerr= [infer_data[column[0]] -
                           infer_data['lower25 '+column[0]],
                           infer_data['upper75 '+column[0]] -
                           infer_data[column[0]]],
                           linestyle='', fmt=".",color= point_color[0],
                           ecolor= ecolor[0], elinewidth=elinewidth,
                            label ="ARGinfer", marker = ".", markersize= markersize)
        ax.errorbar(truth_data[column[0]]+epsilon,
                    weaver_data[weaver_col],
                    yerr= [weaver_data[weaver_col] -
                           weaver_data['lower25 '+weaver_col],
                           weaver_data['upper75 '+weaver_col] -
                           weaver_data[weaver_col]],
                           linestyle='', fmt=".",color= point_color[1],
                            ecolor= ecolor[1], elinewidth=elinewidth,
                            label="ARGweaver",marker = "^", markersize= "2")
            # ax.set_aspect('equal',adjustable="box")
        minimum = np.min((ax.get_xlim(),ax.get_ylim()))
        maximum = np.max((ax.get_xlim(),ax.get_ylim()))
        ax.set_ylabel("Inferred ")
        ax.set_xlabel("True "+ column[0])
        #scientific number
        ax.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
        ax.ticklabel_format(style='sci',scilimits=(-2,2),axis='x')
        # ax.set_yscale('log')
        # ax.set_xscale('log')
        # ax.axis('equal')
        # ax.axis('square')
        ax.plot([minimum, maximum], [minimum, maximum], ls="--",  color= line_color, label ="Truth")
        fig.legend(loc = 'upper left', bbox_to_anchor=(0.10, .95),fancybox=True)#upper right
        figure_name= "subsampleScatter"+column[0]
        plt.savefig(arginfer_path+"/{}.pdf".format(figure_name),
                    bbox_inches='tight', dpi=400)
        plt.close()

if __name__=='__main__':

    # s= Scatter(truth_path= '/data/projects/punim0594/Ali/phd/mcmc_out/sim10L100K/sim_r4',
    #            inferred_path='/data/projects/punim0594/Ali/phd/mcmc_out/ARGinfer/M2/n10L100K_r4',
    #            columns=["branch length", 'total recomb', "ancestral recomb", 'posterior'])
    # s.multi_scatter(CI=True, argweaver= False, coverage = True, R=4)
    # s= Scatter(truth_path = '/data/projects/punim0594/Ali/phd/mcmc_out/sim10L100K/sim_r4',
    #            inferred_path='/data/projects/punim0594/Ali/phd/mcmc_out/aw/r4/n10L100K',
    #            columns=["branch length", "ancestral recomb"])
    # s.multi_scatter(CI=True, argweaver= True, coverage = True, R=4)
    # s= Scatter(truth_path= '/data/projects/punim0594/Ali/phd/mcmc_out/aw/r2/n10L100K',
    #            inferred_path='/data/projects/punim0594/Ali/phd/mcmc_out/ARGinfer/n10L100K_r2',
    #            columns=["branch length", 'total recomb', "ancestral recomb", 'posterior'], std=True)
    # s.multi_std()
    # plot_tmrca(truth_path ='/data/projects/punim0594/Ali/phd/mcmc_out/sim10L100K/sim_r1/true_tmrca15.npy',
    #                arginfer_path='/data/projects/punim0594/Ali/phd/mcmc_out/ARGinfer/M2/n10L100K_r1/out15',
    #                 argweaver_path = '/data/projects/punim0594/Ali/phd/mcmc_out/aw/r1/n10L100K/out15',
    #                inferred_filename='tmrca.h5', CI= 95, R=1)

    plot_interval(true_path= '/data/projects/punim0594/Ali/phd/mcmc_out/sim10L100K/sim_r1',
                    argweaver_path='/data/projects/punim0594/Ali/phd/mcmc_out/aw/r1/n10L100K',
                  arginfer_path='/data/projects/punim0594/Ali/phd/mcmc_out/ARGinfer/M2/n10L100K_r1',
                          columns=["branch length", "ancestral recomb"], R=1)

    # recrate_boxplot(argweaver_general_path='/data/projects/punim0594/Ali/phd/mcmc_out/aw',
    #                 arginfer_general_path='/data/projects/punim0594/Ali/phd/mcmc_out/ARGinfer/M2')

    # plot_allele_age(truth_path ='/data/projects/punim0594/Ali/phd/mcmc_out/sim10L100K/sim_r4/true_allele_age7.h5',
    #                arginfer_path='/data/projects/punim0594/Ali/phd/mcmc_out/ARGinfer/M2/n10L100K_r4/out7',
    #                 argweaver_path = '/data/projects/punim0594/Ali/phd/mcmc_out/aw/r4/n10L100K/out7',
    #                inferred_filename='allele_age.h5', R=4, CI=50)

    # get_summary_tmrca_allele_age(truth_general_path ='/data/projects/punim0594/Ali/phd/mcmc_out/sim10L100K/sim_r1',
    #                 arginfer_general_path='/data/projects/punim0594/Ali/phd/mcmc_out/ARGinfer/M2/n10L100K_r1',
    #                 argweaver_general_path ='/data/projects/punim0594/Ali/phd/mcmc_out/aw/r1/n10L100K',
    #                 replicate=161,
    #                 feature = "allele_age", R=4)
    #
    # compare_infer_weaver(truth_path ='/data/projects/punim0594/Ali/phd/mcmc_out/sim10L100K/sim_r4',
    #                      argweaver_path='/data/projects/punim0594/Ali/phd/mcmc_out/aw/r4/n10L100K',
    #                      arginfer_path='/data/projects/punim0594/Ali/phd/mcmc_out/ARGinfer/M2/n10L100K_r4',
    #                      column = ["ancestral recomb"], sub_sample =50)

    # s= Scatter(truth_path= '/data/projects/punim0594/Ali/phd/mcmc_out/sim10L100K/sim_r1',
    #            inferred_path='/data/projects/punim0594/Ali/phd/mcmc_out/ARGinfer/M2/n10L100K_r1',
    #            inferred2_path='/data/projects/punim0594/Ali/phd/mcmc_out/aw/r1/n10L100K',
    #            columns=["ancestral recomb"], weaver_infer=True)
    # s.ARGinfer_weaver(R=1)

    # s= Scatter(truth_path= '/data/projects/punim0594/Ali/phd/mcmc_out/sim10L100K/sim_r4',
    #            inferred_path='/data/projects/punim0594/Ali/phd/mcmc_out/ARGinfer/M2/n10L100K_r4',
    #            inferred2_path='/data/projects/punim0594/Ali/phd/mcmc_out/aw/r4/n10L100K',
    #            columns=["branch length"], weaver_infer=True)
    # s.ARGinfer_weaver(R=4, both_infer_methods = True)
