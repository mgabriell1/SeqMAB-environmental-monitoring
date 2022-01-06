import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, ConstantKernel, WhiteKernel)
#%matplotlib inline
import warnings
warnings.filterwarnings(action='ignore')


#%% Seq(GP-UCB-SW)
def SEQGPUCBSW_max(slidingWindow, maxValue, samplingTimes, dataset, day, hour, plot):
     
    ## GP details
    xmax = max(samplingTimes) 
    X_ = (samplingTimes/xmax)[:, np.newaxis]
        
    ## GP-upper conficende bound definition
    delta = 0.1
    D = len(X_)
    
    ## GP kernel details
    kernel = ConstantKernel(1.0, (1e-3, 1e3))*Matern(nu=2.5) + \
         WhiteKernel(noise_level_bounds=(1e-10,7.5e-5)) 
            
    # Select data in sliding window for GP estimation 
    trainData_orig = dataset[dataset.Day > day - slidingWindow]
    extraData_early = trainData_orig[trainData_orig.sampledTime < 6]
    extraData_early.sampledTime = extraData_early.sampledTime + 24  
    extraData_late = trainData_orig[trainData_orig.sampledTime > 18]
    extraData_late.sampledTime = extraData_late.sampledTime - 24 
    trainData = pd.concat([trainData_orig,extraData_early,extraData_late])
    xtrain = trainData.sampledTime.to_numpy()/xmax
    Xtrain = xtrain[:, np.newaxis]
    ytrain = trainData.Concentration.to_numpy()/maxValue
 
    # Fit GP to data in current window
    gp = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=5)
    gp.fit(Xtrain, ytrain)
     
    # Estimate UCB and select max
    nPulls = len(xtrain)
    y_mean, y_std = gp.predict(X_, return_std=True)
    beta = 2*np.log10(D*nPulls**2*np.pi**2/(6*delta))             
    y_UCB = np.round(y_mean + beta*y_std, decimals = 7) # Rounded bounds to avoid numerical issues
    y_LCB = np.round(y_mean - beta*y_std, decimals = 7)

    # Select samlping time which shows maximum UCB. First a test on the remaining
    # sampling times in the same day is performed, if none is found then the first
    # available value is taken
    new_x = X_[np.argwhere(np.isclose(y_UCB, np.amax(y_UCB))).ravel()]
    nextSamplingTime = next((x*xmax for x in new_x if x > hour/xmax), None)
    if nextSamplingTime == None:
        new_x = X_[[np.argmax(y_UCB)]]
        nextSamplingTime = new_x.ravel()*xmax
    nextSamplingTime = float(nextSamplingTime)
     
    # Plot posterior gaussian process
    if plot:
        plt.subplot(1, 1, 1)
        plt.plot(X_*xmax, (y_mean)*maxValue, 'k', lw=3, zorder=9)
        plt.fill_between(np.squeeze(X_)*xmax,(y_LCB)*maxValue,(y_UCB)*maxValue, alpha=0.2, color='k')
        plt.scatter(trainData_orig.sampledTime,trainData_orig.Concentration, c='b', s=50, zorder=10, edgecolors=(0, 0, 0))
        plt.scatter(nextSamplingTime,np.max(y_UCB)*maxValue,marker='x',c='r')
        plt.title("Day: %.1f, Hour: %.2f \nNew sampling time: %.2f" % (day, hour, nextSamplingTime), fontsize=12)
        plt.xlabel('Hours')
        plt.ylabel('Concentration')
        plt.ylim(0)
        plt.tight_layout()
        plt.show()
     
    # Estimation of unbiased values
    nMax = np.array([0]*len(np.unique(Xtrain)))
    Xsampled = np.unique(Xtrain)[:,np.newaxis]
    nSamples = 100
    for m in range(nSamples):
        draw = gp.sample_y(Xsampled,random_state = None)
        nMax[np.argmax(draw)] = nMax[np.argmax(draw)] + 1
    nMax = nMax/nSamples
    df_temp = pd.DataFrame({'x' : Xtrain[:,0],'y' : ytrain*maxValue})
    yavg = df_temp.groupby('x').mean().y.to_numpy()
     
    max_est_conc = np.sum(nMax*yavg)
    max_est_hour = Xsampled[np.argmax(nMax)]*xmax
        
    return  nextSamplingTime, max_est_conc, max_est_hour

def SEQGPUCBSW_delta(slidingWindow, maxValue, samplingTimes, dataset, day, hour, plot):
     
    ## GP details
    xmax = max(samplingTimes) 
    X_ = (samplingTimes/xmax)[:, np.newaxis]
        
    ## GP-upper conficende bound definition
    delta = 0.1
    D = len(X_)
    
    ## GP kernel details
    kernel = ConstantKernel(1.0, (1e-3, 1e3))*Matern(nu=2.5) + \
         WhiteKernel(noise_level_bounds=(1e-10,7.5e-5)) 
            
    # Select data in sliding window for GP estimation 
    trainData_orig = dataset[dataset.Day > day - slidingWindow]
    extraData_early = trainData_orig[trainData_orig.sampledTime < 6]
    extraData_early.sampledTime = extraData_early.sampledTime + 24  
    extraData_late = trainData_orig[trainData_orig.sampledTime > 18]
    extraData_late.sampledTime = extraData_late.sampledTime - 24 
    trainData = pd.concat([trainData_orig,extraData_early,extraData_late])
    xtrain = trainData.sampledTime.to_numpy()/xmax
    Xtrain = xtrain[:, np.newaxis]
    ytrain = trainData.Concentration.to_numpy()/maxValue
 
    # Fit GP to data in current window
    gp = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=5)
    gp.fit(Xtrain, ytrain)
     
    # Estimate UCB and select max
    nPulls = len(xtrain)
    y_mean, y_std = gp.predict(X_, return_std=True)
    beta = 2*np.log10(D*nPulls**2*np.pi**2/(6*delta))             
    y_UCB = np.round(y_mean + beta*y_std, decimals = 7) # Rounded bounds to avoid numerical issues
    y_LCB = np.round(y_mean - beta*y_std, decimals = 7)

    # Select samlping time which shows maximum UCB. First a test on the remaining
    # sampling times in the same day is performed, if none is found then the first
    # available value is taken
    y_UCB = np.round(y_mean + beta*y_std,6)
    y_LCB = np.round(y_mean - beta*y_std,6)
    X_after = X_[(X_>hour/xmax).ravel()]
    y_UCB_after = y_UCB[(X_>hour/xmax).ravel()]
    y_LCB_after = y_LCB[(X_>hour/xmax).ravel()]
    if len(X_after) > 0:
        new_x_max = X_after[np.argwhere(y_UCB_after == np.amax(y_UCB)).ravel()].ravel()
        new_x_min = X_after[np.argwhere(y_LCB_after == np.amin(y_LCB)).ravel()].ravel()
    else:
        new_x_max = []
        new_x_min = []
    if len(new_x_max) == 0 or len(X_after) == 0:
        new_x_max = X_[[np.argmax(y_UCB)]]
        new_x_max = new_x_max.ravel()
    if len(new_x_min) == 0 or len(X_after) == 0:
        new_x_min = X_[[np.argmin(y_LCB)]]
        new_x_min = new_x_min.ravel()
    if len(new_x_max) > 0:
        new_x_max = np.array([min(new_x_max)])
    if len(new_x_min) > 0:
        new_x_min = np.array([min(new_x_min)])
        
    nextSamplingTime = np.concatenate([new_x_max*xmax,new_x_min*xmax])
        
    # Plot posterior gaussian process
    if plot:
        plt.subplot(1, 1, 1)
        plt.plot(X_*xmax, (y_mean)*maxValue, 'k', lw=3, zorder=9)
        plt.fill_between(np.squeeze(X_)*xmax,(y_LCB)*maxValue,(y_UCB)*maxValue, alpha=0.2, color='k')
        plt.scatter(trainData_orig.sampledTime,trainData_orig.Concentration, c='b', s=50, zorder=10, edgecolors=(0, 0, 0))
        plt.scatter(nextSamplingTime[0],np.max(y_UCB)*maxValue,marker='x',c='r')
        plt.scatter(nextSamplingTime[1],np.max([0,np.min(y_LCB)*maxValue]),marker='x',c='r')
        plt.title("Day: %.1f, Hour: %.2f \nNew sampling times: %.2f, %.2f" % (day, hour, nextSamplingTime[0], nextSamplingTime[1]), fontsize=12)
        plt.xlabel('Hours')
        plt.ylabel('Concentration')
        plt.ylim(0)
        plt.tight_layout()
        plt.show()
     
    # Estimation of unbiased values
    nMax = np.array([0]*len(np.unique(Xtrain)))
    nMin = np.array([0]*len(np.unique(Xtrain)))
    Xsampled = np.unique(Xtrain)[:,np.newaxis]
    nSamples = 100
    for m in range(nSamples):
        draw = gp.sample_y(Xsampled,random_state = None)
        nMax[np.argmax(draw)] = nMax[np.argmax(draw)] + 1
        nMin[np.argmin(draw)] = nMin[np.argmin(draw)] + 1
    nMax = nMax/nSamples
    nMin = nMin/nSamples
    df_temp = pd.DataFrame({'x' : Xtrain[:,0],'y' : ytrain*maxValue})
    yavg = df_temp.groupby('x').mean().y.to_numpy()
     
    max_est_conc = np.sum(nMax*yavg)
    max_est_hour = Xsampled[np.argmax(nMax)]*xmax
    min_est_conc = np.sum(nMin*yavg)
    min_est_hour = Xsampled[np.argmax(nMin)]*xmax
        
    return  nextSamplingTime, max_est_conc, max_est_hour, min_est_conc, min_est_hour


#%% Seq(GP-UCB-CD)

def SEQGPUCBCD_max(trainingWindow, ARL0, explPerc, CP, maxValue, samplingTimes, dataset, day, hour, plot):
    
    ## GP details
    xmax = max(samplingTimes) 
    X_ = (samplingTimes/xmax)[:, np.newaxis]
        
    ## GP-upper conficende bound definition
    delta = 0.1
    D = len(X_)
    
    ## GP kernel details
    kernel = ConstantKernel(1.0, (1e-3, 1e3))*Matern(nu=2.5) + \
         WhiteKernel(noise_level_bounds=(1e-10,7.5e-5)) 
    
    # Select data after last changepoint for GP estimation 
    trainData_orig = dataset[dataset.Day >= CP]
    extraData_early = trainData_orig[trainData_orig.sampledTime < 6]
    extraData_early.sampledTime = extraData_early.sampledTime + 24  
    extraData_late = trainData_orig[trainData_orig.sampledTime > 18]
    extraData_late.sampledTime = extraData_late.sampledTime - 24 
    trainData = pd.concat([trainData_orig,extraData_early,extraData_late])
    xtrain = trainData.sampledTime.to_numpy()/xmax
    Xtrain = xtrain[:, np.newaxis]
    ytrain = trainData.Concentration.to_numpy()/maxValue
        
    # Fit GP to data in current window
    gp = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=5)
    gp.fit(Xtrain, ytrain)
                    
    # Estimate UCB and select max
    nPulls = len(xtrain)
    y_mean, y_std = gp.predict(X_, return_std=True)
    beta = 2*np.log10(D*nPulls**2*np.pi**2/(6*delta))             
    y_UCB = np.round(y_mean + beta*y_std, decimals = 7) # Rounded bounds to avoid numerical issues
    y_LCB = np.round(y_mean - beta*y_std, decimals = 7)
    # Select samlping time which shows maximum UCB. First a test on the remaining
    # sampling times in the same day is performed, if none is found then the first
    # available value is taken
    new_x = X_[np.argwhere(np.isclose(y_UCB, np.amax(y_UCB))).ravel()]
    nextMax = (x*xmax for x in new_x if x > hour/xmax)
    nextSamplingTime = next(nextMax, None)
    if nextSamplingTime == None:
        new_x = X_[[np.argmax(y_UCB)]]
        nextSamplingTime = new_x.ravel()*xmax
    if np.random.random() < explPerc: # Select a random sampling point with the selected probability
        new_x = X_[[np.random.randint(len(X_))]]
        nextSamplingTime = new_x.ravel()*xmax
    nextSamplingTime = float(nextSamplingTime)

                        
    # Plot posterior gaussian process
    if plot:
        plt.subplot(1, 1, 1)
        plt.plot(X_*xmax, (y_mean)*maxValue, 'k', lw=3, zorder=9)
        plt.fill_between(np.squeeze(X_)*xmax,(y_LCB)*maxValue,(y_UCB)*maxValue, alpha=0.2, color='k')
        plt.scatter(trainData_orig.sampledTime,trainData_orig.Concentration, c='b', s=50, zorder=10, edgecolors=(0, 0, 0))
        plt.scatter(nextSamplingTime,np.max(y_UCB)*maxValue,marker='x',c='r')
        plt.title("Day: %.1f, Hour: %.2f \nNew sampling time: %.2f" % (day, hour, nextSamplingTime), fontsize=12)
        plt.xlabel('Hours')
        plt.ylabel('Concentration')
        plt.ylim(0)
        plt.tight_layout()
        plt.show()
       
    # Estimation of unbiased values
    nMax = np.array([0]*len(np.unique(Xtrain)))
    Xsampled = np.unique(Xtrain)[:,np.newaxis]
    nSamples = 100
    for m in range(nSamples):
        draw = gp.sample_y(Xsampled,random_state = None)
        nMax[np.argmax(draw)] = nMax[np.argmax(draw)] + 1
    nMax = nMax/nSamples
    df_temp = pd.DataFrame({'x' : Xtrain[:,0],'y' : ytrain*maxValue})
    yavg = df_temp.groupby('x').mean().y.to_numpy()
    
    max_est_conc = np.sum(nMax*yavg)
    max_est_hour = float(Xsampled[np.argmax(nMax)])*xmax 
    
    return nextSamplingTime, max_est_conc, max_est_hour


def SEQGPUCBCD_delta(trainingWindow, ARL0, explPerc, CP, maxValue, samplingTimes, dataset, day, hour, plot):
    
    ## GP details
    xmax = max(samplingTimes) 
    X_ = (samplingTimes/xmax)[:, np.newaxis]
        
    ## GP-upper conficende bound definition
    delta = 0.1
    D = len(X_)
    
    ## GP kernel details
    kernel = ConstantKernel(1.0, (1e-3, 1e3))*Matern(nu=2.5) + \
         WhiteKernel(noise_level_bounds=(1e-10,7.5e-5)) 
    
    # Select data after last changepoint for GP estimation 
    trainData_orig = dataset[dataset.Day >= CP]
    extraData_early = trainData_orig[trainData_orig.sampledTime < 6]
    extraData_early.sampledTime = extraData_early.sampledTime + 24  
    extraData_late = trainData_orig[trainData_orig.sampledTime > 18]
    extraData_late.sampledTime = extraData_late.sampledTime - 24 
    trainData = pd.concat([trainData_orig,extraData_early,extraData_late])
    xtrain = trainData.sampledTime.to_numpy()/xmax
    Xtrain = xtrain[:, np.newaxis]
    ytrain = trainData.Concentration.to_numpy()/maxValue
        
    # Fit GP to data in current window
    gp = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=5)
    gp.fit(Xtrain, ytrain)
                    
    # Estimate UCB and select max
    nPulls = len(xtrain)
    y_mean, y_std = gp.predict(X_, return_std=True)
    beta = 2*np.log10(D*nPulls**2*np.pi**2/(6*delta))             
    y_UCB = np.round(y_mean + beta*y_std, decimals = 7) # Rounded bounds to avoid numerical issues
    y_LCB = np.round(y_mean - beta*y_std, decimals = 7)
    # Select samlping time which shows maximum UCB. First a test on the remaining
    # sampling times in the same day is performed, if none is found then the first
    # available value is taken  
    y_UCB = np.round(y_mean + beta*y_std,6)
    y_LCB = np.round(y_mean - beta*y_std,6)
    X_after = X_[(X_>hour/xmax).ravel()]
    y_UCB_after = y_UCB[(X_>hour/xmax).ravel()]
    y_LCB_after = y_LCB[(X_>hour/xmax).ravel()]
    if len(X_after) > 0:
        new_x_max = X_after[np.argwhere(y_UCB_after == np.amax(y_UCB)).ravel()].ravel()
        new_x_min = X_after[np.argwhere(y_LCB_after == np.amin(y_LCB)).ravel()].ravel()
    else:
        new_x_max = []
        new_x_min = []
    if len(new_x_max) == 0 or len(X_after) == 0:
        new_x_max = X_[[np.argmax(y_UCB)]]
        new_x_max = new_x_max.ravel()
    if len(new_x_min) == 0 or len(X_after) == 0:
        new_x_min = X_[[np.argmin(y_LCB)]]
        new_x_min = new_x_min.ravel()
    if len(new_x_max) > 0:
        new_x_max = np.array([min(new_x_max)])
    if len(new_x_min) > 0:
        new_x_min = np.array([min(new_x_min)])
    
    # Select a random sampling point instead of the maximum concentration one with the selected probability
    if np.random.random() < explPerc: 
        new_x_max = X_[[np.random.randint(len(X_))]]
        new_x_max = new_x_max.ravel()
    # Select a random sampling point instead of the minimum concentration one with the selected probability
    if np.random.random() < explPerc: # Select a random sampling point with the selected probability
        new_x_min = X_[[np.random.randint(len(X_))]]
        new_x_min = new_x_min.ravel()
        
    nextSamplingTime = np.concatenate([new_x_max*xmax,new_x_min*xmax])

                        
    # Plot posterior gaussian process
    if plot:
        plt.subplot(1, 1, 1)
        plt.plot(X_*xmax, (y_mean)*maxValue, 'k', lw=3, zorder=9)
        plt.fill_between(np.squeeze(X_)*xmax,(y_LCB)*maxValue,(y_UCB)*maxValue, alpha=0.2, color='k')
        plt.scatter(trainData_orig.sampledTime,trainData_orig.Concentration, c='b', s=50, zorder=10, edgecolors=(0, 0, 0))
        plt.scatter(nextSamplingTime[0],np.max(y_UCB)*maxValue,marker='x',c='r')
        plt.scatter(nextSamplingTime[1],np.max([0,np.min(y_LCB)*maxValue]),marker='x',c='r')
        plt.title("Day: %.1f, Hour: %.2f \nNew sampling times: %.2f, %.2f" % (day, hour, nextSamplingTime[0], nextSamplingTime[1]), fontsize=12)
        plt.xlabel('Hours')
        plt.ylabel('Concentration')
        plt.ylim(0)
        plt.tight_layout()
        plt.show()
       
    # Estimation of unbiased values
    nMax = np.array([0]*len(np.unique(Xtrain)))
    nMin = np.array([0]*len(np.unique(Xtrain)))
    Xsampled = np.unique(Xtrain)[:,np.newaxis]
    nSamples = 100
    for m in range(nSamples):
        draw = gp.sample_y(Xsampled,random_state = None)
        nMax[np.argmax(draw)] = nMax[np.argmax(draw)] + 1
        nMin[np.argmin(draw)] = nMin[np.argmin(draw)] + 1
    nMax = nMax/nSamples
    nMin = nMin/nSamples
    df_temp = pd.DataFrame({'x' : Xtrain[:,0],'y' : ytrain*maxValue})
    yavg = df_temp.groupby('x').mean().y.to_numpy()
     
    max_est_conc = np.sum(nMax*yavg)
    max_est_hour = Xsampled[np.argmax(nMax)]*xmax
    min_est_conc = np.sum(nMin*yavg)
    min_est_hour = Xsampled[np.argmax(nMin)]*xmax
    
    return nextSamplingTime, max_est_conc, max_est_hour, min_est_conc, min_est_hour