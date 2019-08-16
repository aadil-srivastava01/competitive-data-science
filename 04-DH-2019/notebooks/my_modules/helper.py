import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def category_freq(dataframe, colname):
    """
    plots a univariate plot for a categorical variable
    
    input: dataframe and column name
    output: frequency plot
    """
    
    hist = {} # initializing the dictionary
    for i in dataframe[colname]:
        hist[i] = hist.get(i, 0) + 1
        
    # visualize the dictionary
    keys, values = list(hist.keys()), list(hist.values())
    
    my_colors = 'rgbymc' # define the colors
    
    # tick_label works like plt.xticks
    bars = plt.bar(range(len(hist)), 
                   values, 
                   tick_label = keys,
                   color = my_colors)
    
    if len(hist)>5:
        plt.xticks(rotation = 90)
    plt.title(colname + " - frequency per category")
    plt.xlabel(colname)
    plt.ylabel("Frequency")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x(), yval + int(len(dataframe)/500), str(yval))
    plt.show()
    
    
class Resampler(object):
    """Represents a framework for computing sampling distribution"""
    
    def __init__(self, sample, xlim = 100):
        """stores the actual sample"""
        self.sample = sample
        self.n = len(sample)
        self.xlim = xlim
        
    def resample(self):
        """generate a new sample by choosing from the original sample with replacement. If you sample without replacement,
        we will get same dataset over and over again."""
        new_sample = np.random.choice(self.sample, self.n, replace = True)
        return new_sample
    
    def sample_stat(self, sample):
        return sample.mean()
    
    def compute_sampling_distribution(self, iterations = 1000):
        """Simulates many experiments and collects the resulting sample statistics"""
        stats = [self.sample_stat(self.resample()) for i in range(iterations)]
        return np.array(stats)
    
    def plot_sampling_distribution(self):
        """plots the sampling distribution"""
        sample_stats = self.compute_sampling_distribution()
        se = sample_stats.std()
        ci = np.percentile(sample_stats, [5, 95])
        
        sns.distplot(sample_stats, color = "red")
        plt.xlabel("sample statistics")
        plt.xlim(self.xlim)
        
        se_str = "SE = " + str(se)
        ci_str = "CI = " + str(ci)
        
        ax = plt.gca()
        plt.text(0.3, 0.95, s = se_str, horizontalalignment = "center", verticalalignment = "center", transform = ax.transAxes)
        plt.text(0.7, 0.95, s = ci_str, horizontalalignment = "center", verticalalignment = "center", transform = ax.transAxes)
        
        plt.show()
        
def interact_fn(x, n, xlim):
    """
    forms the object of Resampler class
    """
    x = x.values
    sample = np.random.choice(x, n)
    resampler = Resampler(sample, xlim = xlim)
    resampler.plot_sampling_distribution()
    
    
def cohen_effect_size(group1, group2):
    diff = group1.mean() - group2.mean()
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    pooled_var = ((n1*var1) + (n2*var2))/(n1 + n2)
    d = diff/np.sqrt(pooled_var)
    return d

def overlap_superiority(control, treatment, n = 1000):
    """ Estimates overlap and superiority based on a sample
    control: rv object
    treatment: rv object
    n: samples """
    
    control_samples = control.rvs(n)
    treatment_samples = treatment.rvs(n)
    threshold = (control.mean() + treatment.mean())/2
    control_above = sum(control_samples > threshold)
    treatment_below = sum(treatment_samples < threshold)
    overlap = (control_above + treatment_below)/n
    
    superiority = (treatment_samples > control_samples).mean()
    return overlap, superiority

def plot_pdfs(cohen_d):
    """Plot PDFs for distributions that differ by some number of stds.
    
    cohen_d: number of standard deviations between the means
    """
    control = sps.norm(0, 1)
    treatment = sp  s.norm(cohen_d, 1)
    xs, ys = eval_pdf(control)
    plt.fill_between(xs, ys, label='control', color="blue", alpha=0.7)

    xs, ys = eval_pdf(treatment)
    plt.fill_between(xs, ys, label='treatment', color='red', alpha=0.7)
    
    o, s = overlap_superiority(control, treatment)
    plt.text(0, 0.05, 'overlap ' + str(o))
    plt.text(0, 0.15, 'superiority ' + str(s))
    plt.show()