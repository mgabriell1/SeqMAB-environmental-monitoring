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
from rpy2.robjects import FloatVector
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
pandas2ri.activate()
cpm = importr('cpm') # Import R cpm package

from concPatternTTHMS import concPatternTTHMS
from concPatternICC import concPatternICC
from Algorithms import SEQGPUCBCD_delta
from Algorithms import SEQGPUCBCD_max

### Define monitoring campaing details
samplingDuration = 180
samplingTimes = np.arange(24, step = 1,dtype = float) # All possible sampling times. Here for example sampling is possible every round hour
data = pd.DataFrame(columns=['Day', 'sampledTime', 'Concentration']) # Dataframe to store sampled concentrations
nextSamplingTime = 12 # First sampling time. Can be selected randomly
targetValue = "delta" # Possible values "max" or "delta" 

### Algorithm details
plot = True # Plot fitted results or not
trainingWindow = 30
explPerc = 0.075
ARL0 = 500
detectedCP = 0
# Setup the online change detection test
cpModConc_max = cpm.makeChangePointModel(cpmType='Lepage', ARL0=ARL0) 
if targetValue == "delta":
	cpModConc_min = cpm.makeChangePointModel(cpmType='Lepage', ARL0=ARL0) 
	cpModConc_delta = cpm.makeChangePointModel(cpmType='Lepage', ARL0=ARL0) 

# To test the algorithms two synthetic daily patterns (TTHMS/ICC) are provided.
# In both pattern the presence of a changepoint can be simulated
Simulation = 'ICC' # Available synthetic datasets: 'THMS' or 'ICC'

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
    
### Sampling
for i in range(samplingDuration):
    dailyObsConc = []
    dailyObsLoc = []
    for j in samplingTimes:
        if np.any(np.isclose(j, nextSamplingTime)):
            # Sample previously specified sampling time
            if Simulation == 'THMS':
                sampledConc = concPatternTTHMS(j,i,typeCP,dayCP,entityCP)
            if Simulation == 'ICC':
                sampledConc = concPatternICC(j,i,typeCP,dayCP,entityCP) 
            data = data.append({'Day': i, 'sampledTime': j, 'Concentration': sampledConc}, ignore_index = True)
            dailyObsConc = dailyObsConc + [sampledConc]
            dailyObsLoc = dailyObsLoc + [j]
            
            # Select new sampling time
            if targetValue == "max":
            	nextSamplingTime, max_est_conc, max_est_loc  = SEQGPUCBCD_max(trainingWindow, ARL0, explPerc, detectedCP, maxValue, samplingTimes, data, i, j, plot)
            if targetValue == "delta":
            	nextSamplingTime, max_est_conc, max_est_loc, min_est_conc, min_est_loc = SEQGPUCBCD_delta(trainingWindow, ARL0, explPerc, detectedCP, maxValue, samplingTimes, data, i, j, plot)
            
    # After the pattern is characterized (at the end of the training window) variations in the daily maximum value are monitored
    if i - detectedCP >= trainingWindow:
        # Maximum concentration change detection
        cpModConc_max = cpm.processObservation(cpModConc_max, max(dailyObsConc))
        detectionConc_max = cpm.changeDetected(cpModConc_max)
        if 1 in detectionConc_max:
            Ds = cpm.getStatistics(cpModConc_max)
            occurrenceCP = np.argmax(Ds) + detectedCP + trainingWindow
            cpModConc_max = cpm.cpmReset(cpModConc_max)
            if targetValue == "delta":
	            cpModConc_min = cpm.cpmReset(cpModConc_min)
	            cpModConc_delta = cpm.cpmReset(cpModConc_delta)
            print('Change in maximum concentration occurred at ', occurrenceCP,' detected at ', i) 
            detectedCP = i
        
        if targetValue == "delta":
	        # Minimum concentration change detection
	        cpModConc_min = cpm.processObservation(cpModConc_min, min(dailyObsConc))
	        detectionConc_min = cpm.changeDetected(cpModConc_min)
	        if 1 in detectionConc_min:
	            Ds = cpm.getStatistics(cpModConc_min)
	            occurrenceCP = np.argmax(Ds) + detectedCP + trainingWindow
	            cpModConc_max = cpm.cpmReset(cpModConc_max)
	            cpModConc_min = cpm.cpmReset(cpModConc_min)
	            cpModConc_delta = cpm.cpmReset(cpModConc_delta)
	            print('Change in minimum concentration occurred at ', occurrenceCP,' detected at ', i) 
	            detectedCP = i
	        
	        # Maximum concentration variation change detection    
	        cpModConc_delta = cpm.processObservation(cpModConc_delta, max(dailyObsConc) - min(dailyObsConc))
	        detectionConc_delta = cpm.changeDetected(cpModConc_delta)
	        if 1 in detectionConc_delta:
	            Ds = cpm.getStatistics(cpModConc_delta)
	            occurrenceCP = np.argmax(Ds) + detectedCP + trainingWindow
	            cpModConc_delta = cpm.cpmReset(cpModConc_delta)
	            cpModConc_max = cpm.cpmReset(cpModConc_max)
	            print('Change in maximum daily concentration variation occurred at ', occurrenceCP,' detected at ', i) 
	            detectedCP = i

              

