import os
# Set working directory same as file
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
import sys
sys.path.append(dname+'/Environments')
sys.path.append(dname+'/Algorithms')
import pandas as pd
import numpy as np

from concPatternTTHMS import concPatternTTHMS
from concPatternICC import concPatternICC
from Algorithms import SEQGPUCBSW_delta
from Algorithms import SEQGPUCBSW_max


#%% Maximum and minimum concentration sampling

### Define monitoring campaing details
samplingDuration = 180
samplingTimes = np.arange(24, step = 1,dtype = float) # All possible sampling times. Here for example sampling is possible every round hour
data = pd.DataFrame(columns=['Day', 'sampledTime', 'Concentration']) # Dataframe to store sampled concentrations
nextSamplingTime = 12 # First sampling time. Can be selected randomly
targetValue = "max" # Possible values "max" or "delta" 

### Algorithm details
slidingWindow = 25
plot = True # Plot fitted results or not. 
# Blue dots indicate samples included in the sliding window, while red crosses the next sampling instants

# To test the algorithms two synthetic daily patterns (TTHMS/ICC) are provided.
# In both pattern the presence of a changepoint can be simulated    
Simulation = 'THMS' # Available synthetic datasets: 'THMS' or 'ICC'

if Simulation == 'THMS':
    maxValue = 150 # [ppb]  Realistic maximum expected concentration. Used only to aid algorithm fit
    
    # Pattern change details
    typeCP = ['shift'] # 'ampl' OR 'mean' OR 'shift. More than one option can be tested contemporarily
    dayCP = [70,120] # Beginning and end days of the gradual pattern change
    entityCP = [6] # Entity of the selected 'typeCP'. It has to be of the same lenght as 'typeCP'
    
if Simulation == 'ICC':
    maxValue = 500 #[cells/ul]  Realistic maximum expected concentration. Used only to aid algorithm fit
    
    # Pattern change details
    typeCP = ['shift'] # 'ampl' OR 'mean' OR 'shift. More than one option can be tested contemporarily
    dayCP = 90 # Day of abrupt pattern change
    entityCP = [1] # Entity of the selected 'typeCP'. It has to be of the same lenght as 'typeCP'
  # Entity of the selected 'typeCP'. It has to be of the same lenght as 'typeCP'
    
### Sampling
for i in range(samplingDuration):
    for j in samplingTimes:
        if np.any(np.isclose(j, nextSamplingTime)):
            # Sample previously specified sampling time
            if Simulation == 'THMS':
                sampledConc = concPatternTTHMS(j,i,typeCP,dayCP,entityCP)
            if Simulation == 'ICC':
                sampledConc = concPatternICC(j,i,typeCP,dayCP,entityCP) 
            data = data.append({'Day': i, 'sampledTime': j, 'Concentration': sampledConc}, ignore_index = True)
            
            # Select new sampling time
            if targetValue == "max":
                nextSamplingTime, max_est_conc, max_est_loc = SEQGPUCBSW_max(slidingWindow, maxValue, samplingTimes, data, i, j, plot)
            if targetValue == "delta":
                nextSamplingTime, max_est_conc, max_est_loc, min_est_conc, min_est_loc = SEQGPUCBSW_delta(slidingWindow, maxValue, samplingTimes, data, i, j, plot)
            

