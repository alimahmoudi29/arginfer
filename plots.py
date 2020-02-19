import os
import pandas as pd
import numpy as np
import matplotlib
# matplotlib.use('Agg')  # NOQA
import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition, inset_axes
import seaborn as sns
import scipy
import time

class Figure(object):
    """
    Superclass of figures . Each figure is a concrete subclass.
    """
    name = None

    def __init__(self, outpath = os.getcwd() +"/output"):
        self.outpath = outpath
        datafile_name = self.outpath + "/{}.h5".format(self.name)
        self.data = pd.read_hdf(datafile_name, mode="r")

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

class plot_summary(Figure):

    name = "summary"

    def plot(self,  true_values= True):
        df = self.data
        truth =  self.load_true_values()
        true_anc_recomb= truth[3]
        true_nonanc_rec = truth [4]
        true_branch_length = truth[5]
        fig = plt.figure()
        fig.subplots_adjust(hspace = 0.4, wspace = 0.4)
        for i,  d in zip(range(6), ["likelihood", "prior", "posterior","branch length",
                                    "ancestral recomb", "non ancestral recomb"]):
            fig.add_subplot(3, 2, i+1)
            df = self.data[d]
            plt.plot(df)
            if true_values:
                plt.axhline(y= truth[i], color="r", linestyle = "--", lw= 1)
            plt.ylabel(d)
            if i>3:
                plt.xlabel("Iteration")
        fig.suptitle("Iter = " +str(int(self.data.setup[0]/1000)) + "K "+", thin = "+\
            str(int(self.data.setup[1]))+ " "+", burn: "+ str(int(self.data.setup[2]))+\
            ", n= " + str(int(self.data.setup[3]))+", Ne = "+ str(int(self.data.setup[6]/1000))  +\
                     "K,\n L= "+ str(int(self.data.setup[4]/1000))+\
            "K, m= " + str(int(self.data.setup[5]))+ ", accept= "+ str(self.data.setup[9])+\
            ", CPU time = " + str(int(self.data.setup[10]/60))+ " min\n" +
                     "detail accept: ["+ str(self.data.setup[11]) +", " + str(self.data.setup[12])+ ", " +
                     str(self.data.setup[13]) + ", "+ str(self.data.setup[14])+ ", "
                     + str(self.data.setup[15])+ " ,"+ str(self.data.setup[16]) +"]")
        self.save(figure_name="summary" + time.strftime("%Y%m%d-%H%M%S"))
        plt.show()




