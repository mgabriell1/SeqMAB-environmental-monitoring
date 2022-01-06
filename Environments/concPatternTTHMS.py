#######
## TTHM daily pattern
## from Chaib and Moschandreas (2008) doi:10.1016/j.jhazmat.2007.06.049 
######

import numpy as np

## Concentration pattern with abrupt change in concentrations. Could be simulated
## by just imposing a very fast gradual change
# def concPatternTTHMS(time, day, whatCP, dayCP, entityCP):
          
#     TTHMmean = 98 #ppb
#     TTHMsd = 0  
        
#     AMPLmean = 14.37 #ppb
#     AMPLsd = 3.14
    
#     # TTHMS period
#     Pmean = 24.145 #h
#     Psd = 0.955
    
#     # Shift
#     shift = 5
    
#     # Changepoint
#     if 'mean' in whatCP and day > dayCP:
#         j = whatCP.index('mean')
#         TTHMmean = TTHMmean + entityCP[j]
    
#     if 'ampl' in whatCP and day > dayCP:
#         j = whatCP.index('ampl')
#         AMPLmean = AMPLmean + entityCP[j]
    
#     if 'shift' in whatCP and day > dayCP:
#         j = whatCP.index('shift')
#         shift = shift + entityCP[j]
    
#     TTHM = np.random.normal(TTHMmean,TTHMsd)
#     AMPL = np.random.normal(AMPLmean,AMPLsd)
#     P = np.random.normal(Pmean,Psd)
#     y = TTHM + AMPL*np.sin(2*np.pi*(time-shift)/P)

#     return y

def concPatternTTHMS(time, day, whatCP, dayCP, entityCP):       
    TTHMmean = 98 #ppb
    TTHMsd = 0  
        
    AMPLmean = 14.37 #ppb
    AMPLsd = 3.14
    
    # TTHMS period
    Pmean = 24.145 #h
    Psd = 0.955
    
    # Shift
    shift = 5
    
    # Gradual change
    if 'mean' in whatCP and day > dayCP[0] and day < dayCP[1]:
        j = whatCP.index('mean')
        TTHMmean = TTHMmean + entityCP[j]*(day-dayCP[0])/(dayCP[1]-dayCP[0])
    
    if 'ampl' in whatCP and day > dayCP[0] and day < dayCP[1]:
        j = whatCP.index('ampl')
        AMPLmean = AMPLmean + entityCP[j]*(day-dayCP[0])/(dayCP[1]-dayCP[0])
    
    if 'shift' in whatCP and day > dayCP[0] and day < dayCP[1]:
        j = whatCP.index('shift')
        shift = shift + entityCP[j]*(day-dayCP[0])/(dayCP[1]-dayCP[0])
    
    
    if 'mean' in whatCP and day > dayCP[1]:
        j = whatCP.index('mean')
        TTHMmean = TTHMmean + entityCP[j]
    
    if 'ampl' in whatCP and day > dayCP[1]:
        j = whatCP.index('ampl')
        AMPLmean = AMPLmean + entityCP[j]
    
    if 'shift' in whatCP and day > dayCP[1]:
        j = whatCP.index('shift')
        shift = shift + entityCP[j]
    
    TTHM = np.random.normal(TTHMmean,TTHMsd)
    AMPL = np.random.normal(AMPLmean,AMPLsd)
    P = np.random.normal(Pmean,Psd)
    y = TTHM + AMPL*np.sin(2*np.pi*(time-shift)/P)
    
    return y

