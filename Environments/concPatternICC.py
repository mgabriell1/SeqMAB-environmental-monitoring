#######
## ICC daily pattern 
## gaussian model derived from Nescerecka et al. (2014) doi: 10.1371/journal.pone.0096354
######

import numpy as np
from scipy.stats import norm


def concPatternICC(time, day, whatCP, dayCP, entityCP):
    
    uncertainty = 0.09
    
    gaus1_mean = 6.71
    gaus1_sd = 1.49
    gaus1_mult = 975.88
    gaus2_mean = 12.83
    gaus2_sd = 0.78
    gaus2_mult = 641.27
    baseline = 181.92
    
    # Changepoint
    if 'mean' in whatCP and day > dayCP:
        j = whatCP.index('mean')
        baseline = baseline + entityCP[j]
    
   
    if 'shift' in whatCP and day > dayCP:
        j = whatCP.index('shift')
        gaus1_mean = gaus1_mean + entityCP[j]
        gaus2_mean = gaus2_mean + entityCP[j]
    

    dat_temp = baseline + gaus1_mult*norm.pdf(time,gaus1_mean,gaus1_sd) + gaus2_mult*norm.pdf(time,gaus2_mean,gaus2_sd)
    y = np.random.normal(dat_temp,dat_temp*uncertainty)
    
    return y



