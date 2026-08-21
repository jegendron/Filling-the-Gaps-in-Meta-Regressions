#%%###########################################################################
###                     DATA SETUP - (Find True Parameters)                ###
##############################################################################

### Audio feedback signals Monte Carlo loop is done
# pip install chime
#import chime
#chime.theme('pokemon')
#chime.theme('material')
#chime.theme('random')

import numpy as np
import pandas as pd
import math
import random
import statsmodels.formula.api as smf
from linearmodels.panel import RandomEffects
#from scipy import stats
#import statsmodels.api as sm
#import time
#from scipy.stats import multivariate_normal

### Start by setting up variance covariance matrix (presumed the same for all studies)

# matrix syntax: (TL, BL, TR, BR)
S00 = np.array([[1, 0.5], [0.5, 1]])                    # top left
S01 = np.array([[0.25, 0], [0.2, 0.1]])                 # top right
S10 = S01.T                                              # bottom left
S11 = np.array([[1, 0.1], [0.1, 1]])                    # bottom right

SIGMA = np.block([[S00, S01], [S10, S11]])               # VarCov matrix

# Elements used to calculate the true parameters
varY = S00[0, 0]
covY = SIGMA[1:len(SIGMA[0, :]), 0:1]
covX = SIGMA[1:len(SIGMA[0, :]), 1:len(SIGMA[0, :])]
  # the length() extracts the length of one row
trueBetas = np.linalg.solve(covX, covY)

### The true parameters
sig2True = varY - covY.T @ trueBetas                     # Omega
b1True = trueBetas[0, 0]
b2True = trueBetas[1, 0]
b3True = trueBetas[2, 0]

R2True = 1 - sig2True/varY  #Not used due to issues with how R2 is measured for fixed vs random vs mixed effects

##############################################################################
###                       DATA SETUP - (Generate the data)                 ###
##############################################################################

#HOW THE DATA IS GENERATED
  #-Each (country) has 1 study ran per year for 5 years
  #-Heterogeneity in countries is captured by the different mean value for each country
  #-Heterogeneity in time is captured by the trend in y=y+0.5*i

# Note: tidyverse functionality will be handled by pandas

# Means
mu_x1 = 1
mu_x2 = 0.5
mu_x3 = 1.5

numYears = 5

#####################################
### SPECIFY SIMULATION PARAMETERS ###
#####################################

# Monte Carlo iterations
#N = 10000
#
N = 1000
#N = 10
#N = 3

# Sample size per study
#n = 50
#
n = 100
#n = 150

numCountries = 15
case = 1  # Options: 1-6

########################################################################################

timeHet = 0
spreadMuY = 0

# timeHet <- Will time heterog be small or large?
# spreadMuY <- Will muY have a large variation (-10 to 10), or small variation (-2 to 2)?

#%%###########################################################################
###                         VARIABLES (for MONTE CARLO)                    ###
##############################################################################

# cca
RE_cca_b0Var = np.zeros(N) # varb0
FE_cca_b0Var = np.zeros(N)
FEt_cca_b0Var = np.zeros(N)
RE_cca_b1Var = np.zeros(N) # varb1
FE_cca_b1Var = np.zeros(N)
FEt_cca_b1Var = np.zeros(N)
RE_cca_b2Var = np.zeros(N) # varb2
FE_cca_b2Var = np.zeros(N)
FEt_cca_b2Var = np.zeros(N)
RE_cca_b3Var = np.zeros(N) # varb3
FE_cca_b3Var = np.zeros(N)
FEt_cca_b3Var = np.zeros(N)
RE_cca_b0 = np.zeros(N) # b0
FE_cca_b0 = np.zeros(N)
FEt_cca_b0 = np.zeros(N)
RE_cca_b1 = np.zeros(N) # b1
FE_cca_b1 = np.zeros(N)
FEt_cca_b1 = np.zeros(N)
RE_cca_b2 = np.zeros(N) # b2
FE_cca_b2 = np.zeros(N)
FEt_cca_b2 = np.zeros(N)
RE_cca_b3 = np.zeros(N) # b3
FE_cca_b3 = np.zeros(N)
FEt_cca_b3 = np.zeros(N)
RE_cca_sig2 = np.zeros(N) # sig2
FE_cca_sig2 = np.zeros(N)
FEt_cca_sig2 = np.zeros(N)
RE_cca_b0SE = np.zeros(N) # SE0
FE_cca_b0SE = np.zeros(N)
FEt_cca_b0SE = np.zeros(N)
RE_cca_b1SE = np.zeros(N) # SE1
FE_cca_b1SE = np.zeros(N)
FEt_cca_b1SE = np.zeros(N)
RE_cca_b2SE = np.zeros(N) # SE2
FE_cca_b2SE = np.zeros(N)
FEt_cca_b2SE = np.zeros(N) 
RE_cca_b3SE = np.zeros(N) # SE3
FE_cca_b3SE = np.zeros(N)
FEt_cca_b3SE = np.zeros(N)
RE_cca_r2 = np.zeros(N) # r2
FE_cca_r2 = np.zeros(N)
FEt_cca_r2 = np.zeros(N)
RE_cca_a_r2 = np.zeros(N) # ar2
FE_cca_a_r2 = np.zeros(N)
FEt_cca_a_r2 = np.zeros(N)
RE_cca_aic = np.zeros(N) # aic
FE_cca_aic = np.zeros(N)
FEt_cca_aic = np.zeros(N)
RE_cca_mse = np.zeros(N) # mse_y
FE_cca_mse = np.zeros(N)
FEt_cca_mse = np.zeros(N)
RE_cca_mae = np.zeros(N) # mae_y
FE_cca_mae = np.zeros(N)
FEt_cca_mae = np.zeros(N)
RE_cca_mpe = np.zeros(N) # mpe_y
FE_cca_mpe = np.zeros(N)
FEt_cca_mpe = np.zeros(N)
RE_cca_mape = np.zeros(N) # mape_y
FE_cca_mape = np.zeros(N)
FEt_cca_mape = np.zeros(N)
RE_cca_mse_x1 = np.zeros(N) # mse_x1
FE_cca_mse_x1 = np.zeros(N)
FEt_cca_mse_x1 = np.zeros(N) 
RE_cca_mae_x1 = np.zeros(N) # mae_x1
FE_cca_mae_x1 = np.zeros(N)
FEt_cca_mae_x1 = np.zeros(N)
RE_cca_mpe_x1 = np.zeros(N) # mpe_x1
FE_cca_mpe_x1 = np.zeros(N)
FEt_cca_mpe_x1 = np.zeros(N)
RE_cca_mape_x1 = np.zeros(N) # mape_x1
FE_cca_mape_x1 = np.zeros(N)
FEt_cca_mape_x1 = np.zeros(N)
RE_cca_mse_x2 = np.zeros(N) # mse_x2
FE_cca_mse_x2 = np.zeros(N)
FEt_cca_mse_x2 = np.zeros(N)
RE_cca_mae_x2 = np.zeros(N) # mae_x2
FE_cca_mae_x2 = np.zeros(N)
FEt_cca_mae_x2 = np.zeros(N) 
RE_cca_mpe_x2 = np.zeros(N) # mpe_x2
FE_cca_mpe_x2 = np.zeros(N)
FEt_cca_mpe_x2 = np.zeros(N)
RE_cca_mape_x2 = np.zeros(N) # mape_x2
FE_cca_mape_x2 = np.zeros(N)
FEt_cca_mape_x2 = np.zeros(N)
RE_cca_mse_x3 = np.zeros(N) # mse_x3
FE_cca_mse_x3 = np.zeros(N)
FEt_cca_mse_x3 = np.zeros(N)
RE_cca_mae_x3 = np.zeros(N) # mae_x2
FE_cca_mae_x3 = np.zeros(N)
FEt_cca_mae_x3 = np.zeros(N)
RE_cca_mpe_x3 = np.zeros(N) # mpe_x2
FE_cca_mpe_x3 = np.zeros(N)
FEt_cca_mpe_x3 = np.zeros(N)
RE_cca_mape_x3 = np.zeros(N) # mape_x2
FE_cca_mape_x3 = np.zeros(N)
FEt_cca_mape_x3 = np.zeros(N)
CIlo_RE_cca_b0 = np.zeros(N) # CI_b0
CIlo_FE_cca_b0 = np.zeros(N)
CIlo_FEt_cca_b0 = np.zeros(N)
CIhi_RE_cca_b0 = np.zeros(N)
CIhi_FE_cca_b0 = np.zeros(N)
CIhi_FEt_cca_b0 = np.zeros(N)
CIlo_RE_cca_b1 = np.zeros(N) # CI_b1
CIlo_FE_cca_b1 = np.zeros(N)
CIlo_FEt_cca_b1 = np.zeros(N)
CIhi_RE_cca_b1 = np.zeros(N)
CIhi_FE_cca_b1 = np.zeros(N)
CIhi_FEt_cca_b1 = np.zeros(N)
CIlo_RE_cca_b2 = np.zeros(N) # CI_b2
CIlo_FE_cca_b2 = np.zeros(N)
CIlo_FEt_cca_b2 = np.zeros(N)
CIhi_RE_cca_b2 = np.zeros(N)
CIhi_FE_cca_b2 = np.zeros(N)
CIhi_FEt_cca_b2 = np.zeros(N)
CIlo_RE_cca_b3 = np.zeros(N) # CI_b3
CIlo_FE_cca_b3 = np.zeros(N)
CIlo_FEt_cca_b3 = np.zeros(N)
CIhi_RE_cca_b3 = np.zeros(N)
CIhi_FE_cca_b3 = np.zeros(N)
CIhi_FEt_cca_b3 = np.zeros(N)

# mi
RE_mi_b0Var = np.zeros(N) # varb0
FE_mi_b0Var = np.zeros(N)
FEt_mi_b0Var = np.zeros(N)
RE_mi_b1Var = np.zeros(N) # varb1
FE_mi_b1Var = np.zeros(N)
FEt_mi_b1Var = np.zeros(N)
RE_mi_b2Var = np.zeros(N) # varb2
FE_mi_b2Var = np.zeros(N)
FEt_mi_b2Var = np.zeros(N)
RE_mi_b3Var = np.zeros(N) # varb3
FE_mi_b3Var = np.zeros(N)
FEt_mi_b3Var = np.zeros(N)
RE_mi_b0 = np.zeros(N) # b0
FE_mi_b0 = np.zeros(N)
FEt_mi_b0 = np.zeros(N)
RE_mi_b1 = np.zeros(N) # b1
FE_mi_b1 = np.zeros(N)
FEt_mi_b1 = np.zeros(N)
RE_mi_b2 = np.zeros(N) # b2
FE_mi_b2 = np.zeros(N)
FEt_mi_b2 = np.zeros(N)
RE_mi_b3 = np.zeros(N) # b3
FE_mi_b3 = np.zeros(N)
FEt_mi_b3 = np.zeros(N)
RE_mi_sig2 = np.zeros(N) # sig2
FE_mi_sig2 = np.zeros(N)
FEt_mi_sig2 = np.zeros(N)
RE_mi_b0SE = np.zeros(N) # SE0
FE_mi_b0SE = np.zeros(N)
FEt_mi_b0SE = np.zeros(N)
RE_mi_b1SE = np.zeros(N) # SE1
FE_mi_b1SE = np.zeros(N)
FEt_mi_b1SE = np.zeros(N)
RE_mi_b2SE = np.zeros(N) # SE2
FE_mi_b2SE = np.zeros(N)
FEt_mi_b2SE = np.zeros(N) 
RE_mi_b3SE = np.zeros(N) # SE3
FE_mi_b3SE = np.zeros(N)
FEt_mi_b3SE = np.zeros(N)
RE_mi_r2 = np.zeros(N) # r2
FE_mi_r2 = np.zeros(N)
FEt_mi_r2 = np.zeros(N)
RE_mi_a_r2 = np.zeros(N) # ar2
FE_mi_a_r2 = np.zeros(N)
FEt_mi_a_r2 = np.zeros(N)
RE_mi_aic = np.zeros(N) # aic
FE_mi_aic = np.zeros(N)
FEt_mi_aic = np.zeros(N)
RE_mi_mse = np.zeros(N) # mse_y
FE_mi_mse = np.zeros(N)
FEt_mi_mse = np.zeros(N)
RE_mi_mae = np.zeros(N) # mae_y
FE_mi_mae = np.zeros(N)
FEt_mi_mae = np.zeros(N)
RE_mi_mpe = np.zeros(N) # mpe_y
FE_mi_mpe = np.zeros(N)
FEt_mi_mpe = np.zeros(N)
RE_mi_mape = np.zeros(N) # mape_y
FE_mi_mape = np.zeros(N)
FEt_mi_mape = np.zeros(N)
RE_mi_mse_x1 = np.zeros(N) # mse_x1
FE_mi_mse_x1 = np.zeros(N)
FEt_mi_mse_x1 = np.zeros(N) 
RE_mi_mae_x1 = np.zeros(N) # mae_x1
FE_mi_mae_x1 = np.zeros(N)
FEt_mi_mae_x1 = np.zeros(N)
RE_mi_mpe_x1 = np.zeros(N) # mpe_x1
FE_mi_mpe_x1 = np.zeros(N)
FEt_mi_mpe_x1 = np.zeros(N)
RE_mi_mape_x1 = np.zeros(N) # mape_x1
FE_mi_mape_x1 = np.zeros(N)
FEt_mi_mape_x1 = np.zeros(N)
RE_mi_mse_x2 = np.zeros(N) # mse_x2
FE_mi_mse_x2 = np.zeros(N)
FEt_mi_mse_x2 = np.zeros(N)
RE_mi_mae_x2 = np.zeros(N) # mae_x2
FE_mi_mae_x2 = np.zeros(N)
FEt_mi_mae_x2 = np.zeros(N) 
RE_mi_mpe_x2 = np.zeros(N) # mpe_x2
FE_mi_mpe_x2 = np.zeros(N)
FEt_mi_mpe_x2 = np.zeros(N)
RE_mi_mape_x2 = np.zeros(N) # mape_x2
FE_mi_mape_x2 = np.zeros(N)
FEt_mi_mape_x2 = np.zeros(N)
RE_mi_mse_x3 = np.zeros(N) # mse_x3
FE_mi_mse_x3 = np.zeros(N)
FEt_mi_mse_x3 = np.zeros(N)
RE_mi_mae_x3 = np.zeros(N) # mae_x2
FE_mi_mae_x3 = np.zeros(N)
FEt_mi_mae_x3 = np.zeros(N)
RE_mi_mpe_x3 = np.zeros(N) # mpe_x2
FE_mi_mpe_x3 = np.zeros(N)
FEt_mi_mpe_x3 = np.zeros(N)
RE_mi_mape_x3 = np.zeros(N) # mape_x2
FE_mi_mape_x3 = np.zeros(N)
FEt_mi_mape_x3 = np.zeros(N)
CIlo_RE_mi_b0 = np.zeros(N) # CI_b0
CIlo_FE_mi_b0 = np.zeros(N)
CIlo_FEt_mi_b0 = np.zeros(N)
CIhi_RE_mi_b0 = np.zeros(N)
CIhi_FE_mi_b0 = np.zeros(N)
CIhi_FEt_mi_b0 = np.zeros(N)
CIlo_RE_mi_b1 = np.zeros(N) # CI_b1
CIlo_FE_mi_b1 = np.zeros(N)
CIlo_FEt_mi_b1 = np.zeros(N)
CIhi_RE_mi_b1 = np.zeros(N)
CIhi_FE_mi_b1 = np.zeros(N)
CIhi_FEt_mi_b1 = np.zeros(N)
CIlo_RE_mi_b2 = np.zeros(N) # CI_b2
CIlo_FE_mi_b2 = np.zeros(N)
CIlo_FEt_mi_b2 = np.zeros(N)
CIhi_RE_mi_b2 = np.zeros(N)
CIhi_FE_mi_b2 = np.zeros(N)
CIhi_FEt_mi_b2 = np.zeros(N)
CIlo_RE_mi_b3 = np.zeros(N) # CI_b3
CIlo_FE_mi_b3 = np.zeros(N)
CIlo_FEt_mi_b3 = np.zeros(N)
CIhi_RE_mi_b3 = np.zeros(N)
CIhi_FE_mi_b3 = np.zeros(N)
CIhi_FEt_mi_b3 = np.zeros(N)

# lh
RE_lh_b0Var = np.zeros(N) # varb0
FE_lh_b0Var = np.zeros(N)
FEt_lh_b0Var = np.zeros(N)
RE_lh_b1Var = np.zeros(N) # varb1
FE_lh_b1Var = np.zeros(N)
FEt_lh_b1Var = np.zeros(N)
RE_lh_b2Var = np.zeros(N) # varb2
FE_lh_b2Var = np.zeros(N)
FEt_lh_b2Var = np.zeros(N)
RE_lh_b3Var = np.zeros(N) # varb3
FE_lh_b3Var = np.zeros(N)
FEt_lh_b3Var = np.zeros(N)
RE_lh_b0 = np.zeros(N) # b0
FE_lh_b0 = np.zeros(N)
FEt_lh_b0 = np.zeros(N)
RE_lh_b1 = np.zeros(N) # b1
FE_lh_b1 = np.zeros(N)
FEt_lh_b1 = np.zeros(N)
RE_lh_b2 = np.zeros(N) # b2
FE_lh_b2 = np.zeros(N)
FEt_lh_b2 = np.zeros(N)
RE_lh_b3 = np.zeros(N) # b3
FE_lh_b3 = np.zeros(N)
FEt_lh_b3 = np.zeros(N)
RE_lh_sig2 = np.zeros(N) # sig2
FE_lh_sig2 = np.zeros(N)
FEt_lh_sig2 = np.zeros(N)
RE_lh_b0SE = np.zeros(N) # SE0
FE_lh_b0SE = np.zeros(N)
FEt_lh_b0SE = np.zeros(N)
RE_lh_b1SE = np.zeros(N) # SE1
FE_lh_b1SE = np.zeros(N)
FEt_lh_b1SE = np.zeros(N)
RE_lh_b2SE = np.zeros(N) # SE2
FE_lh_b2SE = np.zeros(N)
FEt_lh_b2SE = np.zeros(N) 
RE_lh_b3SE = np.zeros(N) # SE3
FE_lh_b3SE = np.zeros(N)
FEt_lh_b3SE = np.zeros(N)
RE_lh_r2 = np.zeros(N) # r2
FE_lh_r2 = np.zeros(N)
FEt_lh_r2 = np.zeros(N)
RE_lh_a_r2 = np.zeros(N) # ar2
FE_lh_a_r2 = np.zeros(N)
FEt_lh_a_r2 = np.zeros(N)
RE_lh_aic = np.zeros(N) # aic
FE_lh_aic = np.zeros(N)
FEt_lh_aic = np.zeros(N)
RE_lh_mse = np.zeros(N) # mse_y
FE_lh_mse = np.zeros(N)
FEt_lh_mse = np.zeros(N)
RE_lh_mae = np.zeros(N) # mae_y
FE_lh_mae = np.zeros(N)
FEt_lh_mae = np.zeros(N)
RE_lh_mpe = np.zeros(N) # mpe_y
FE_lh_mpe = np.zeros(N)
FEt_lh_mpe = np.zeros(N)
RE_lh_mape = np.zeros(N) # mape_y
FE_lh_mape = np.zeros(N)
FEt_lh_mape = np.zeros(N)
RE_lh_mse_x1 = np.zeros(N) # mse_x1
FE_lh_mse_x1 = np.zeros(N)
FEt_lh_mse_x1 = np.zeros(N) 
RE_lh_mae_x1 = np.zeros(N) # mae_x1
FE_lh_mae_x1 = np.zeros(N)
FEt_lh_mae_x1 = np.zeros(N)
RE_lh_mpe_x1 = np.zeros(N) # mpe_x1
FE_lh_mpe_x1 = np.zeros(N)
FEt_lh_mpe_x1 = np.zeros(N)
RE_lh_mape_x1 = np.zeros(N) # mape_x1
FE_lh_mape_x1 = np.zeros(N)
FEt_lh_mape_x1 = np.zeros(N)
RE_lh_mse_x2 = np.zeros(N) # mse_x2
FE_lh_mse_x2 = np.zeros(N)
FEt_lh_mse_x2 = np.zeros(N)
RE_lh_mae_x2 = np.zeros(N) # mae_x2
FE_lh_mae_x2 = np.zeros(N)
FEt_lh_mae_x2 = np.zeros(N) 
RE_lh_mpe_x2 = np.zeros(N) # mpe_x2
FE_lh_mpe_x2 = np.zeros(N)
FEt_lh_mpe_x2 = np.zeros(N)
RE_lh_mape_x2 = np.zeros(N) # mape_x2
FE_lh_mape_x2 = np.zeros(N)
FEt_lh_mape_x2 = np.zeros(N)
RE_lh_mse_x3 = np.zeros(N) # mse_x3
FE_lh_mse_x3 = np.zeros(N)
FEt_lh_mse_x3 = np.zeros(N)
RE_lh_mae_x3 = np.zeros(N) # mae_x2
FE_lh_mae_x3 = np.zeros(N)
FEt_lh_mae_x3 = np.zeros(N)
RE_lh_mpe_x3 = np.zeros(N) # mpe_x2
FE_lh_mpe_x3 = np.zeros(N)
FEt_lh_mpe_x3 = np.zeros(N)
RE_lh_mape_x3 = np.zeros(N) # mape_x2
FE_lh_mape_x3 = np.zeros(N)
FEt_lh_mape_x3 = np.zeros(N)
CIlo_RE_lh_b0 = np.zeros(N) # CI_b0
CIlo_FE_lh_b0 = np.zeros(N)
CIlo_FEt_lh_b0 = np.zeros(N)
CIhi_RE_lh_b0 = np.zeros(N)
CIhi_FE_lh_b0 = np.zeros(N)
CIhi_FEt_lh_b0 = np.zeros(N)
CIlo_RE_lh_b1 = np.zeros(N) # CI_b1
CIlo_FE_lh_b1 = np.zeros(N)
CIlo_FEt_lh_b1 = np.zeros(N)
CIhi_RE_lh_b1 = np.zeros(N)
CIhi_FE_lh_b1 = np.zeros(N)
CIhi_FEt_lh_b1 = np.zeros(N)
CIlo_RE_lh_b2 = np.zeros(N) # CI_b2
CIlo_FE_lh_b2 = np.zeros(N)
CIlo_FEt_lh_b2 = np.zeros(N)
CIhi_RE_lh_b2 = np.zeros(N)
CIhi_FE_lh_b2 = np.zeros(N)
CIhi_FEt_lh_b2 = np.zeros(N)
CIlo_RE_lh_b3 = np.zeros(N) # CI_b3
CIlo_FE_lh_b3 = np.zeros(N)
CIlo_FEt_lh_b3 = np.zeros(N)
CIhi_RE_lh_b3 = np.zeros(N)
CIhi_FE_lh_b3 = np.zeros(N)
CIhi_FEt_lh_b3 = np.zeros(N)

# rf
RE_rf_b0Var = np.zeros(N) # varb0
FE_rf_b0Var = np.zeros(N)
FEt_rf_b0Var = np.zeros(N)
RE_rf_b1Var = np.zeros(N) # varb1
FE_rf_b1Var = np.zeros(N)
FEt_rf_b1Var = np.zeros(N)
RE_rf_b2Var = np.zeros(N) # varb2
FE_rf_b2Var = np.zeros(N)
FEt_rf_b2Var = np.zeros(N)
RE_rf_b3Var = np.zeros(N) # varb3
FE_rf_b3Var = np.zeros(N)
FEt_rf_b3Var = np.zeros(N)
RE_rf_b0 = np.zeros(N) # b0
FE_rf_b0 = np.zeros(N)
FEt_rf_b0 = np.zeros(N)
RE_rf_b1 = np.zeros(N) # b1
FE_rf_b1 = np.zeros(N)
FEt_rf_b1 = np.zeros(N)
RE_rf_b2 = np.zeros(N) # b2
FE_rf_b2 = np.zeros(N)
FEt_rf_b2 = np.zeros(N)
RE_rf_b3 = np.zeros(N) # b3
FE_rf_b3 = np.zeros(N)
FEt_rf_b3 = np.zeros(N)
RE_rf_sig2 = np.zeros(N) # sig2
FE_rf_sig2 = np.zeros(N)
FEt_rf_sig2 = np.zeros(N)
RE_rf_b0SE = np.zeros(N) # SE0
FE_rf_b0SE = np.zeros(N)
FEt_rf_b0SE = np.zeros(N)
RE_rf_b1SE = np.zeros(N) # SE1
FE_rf_b1SE = np.zeros(N)
FEt_rf_b1SE = np.zeros(N)
RE_rf_b2SE = np.zeros(N) # SE2
FE_rf_b2SE = np.zeros(N)
FEt_rf_b2SE = np.zeros(N) 
RE_rf_b3SE = np.zeros(N) # SE3
FE_rf_b3SE = np.zeros(N)
FEt_rf_b3SE = np.zeros(N)
RE_rf_r2 = np.zeros(N) # r2
FE_rf_r2 = np.zeros(N)
FEt_rf_r2 = np.zeros(N)
RE_rf_a_r2 = np.zeros(N) # ar2
FE_rf_a_r2 = np.zeros(N)
FEt_rf_a_r2 = np.zeros(N)
RE_rf_aic = np.zeros(N) # aic
FE_rf_aic = np.zeros(N)
FEt_rf_aic = np.zeros(N)
RE_rf_mse = np.zeros(N) # mse_y
FE_rf_mse = np.zeros(N)
FEt_rf_mse = np.zeros(N)
RE_rf_mae = np.zeros(N) # mae_y
FE_rf_mae = np.zeros(N)
FEt_rf_mae = np.zeros(N)
RE_rf_mpe = np.zeros(N) # mpe_y
FE_rf_mpe = np.zeros(N)
FEt_rf_mpe = np.zeros(N)
RE_rf_mape = np.zeros(N) # mape_y
FE_rf_mape = np.zeros(N)
FEt_rf_mape = np.zeros(N)
RE_rf_mse_x1 = np.zeros(N) # mse_x1
FE_rf_mse_x1 = np.zeros(N)
FEt_rf_mse_x1 = np.zeros(N) 
RE_rf_mae_x1 = np.zeros(N) # mae_x1
FE_rf_mae_x1 = np.zeros(N)
FEt_rf_mae_x1 = np.zeros(N)
RE_rf_mpe_x1 = np.zeros(N) # mpe_x1
FE_rf_mpe_x1 = np.zeros(N)
FEt_rf_mpe_x1 = np.zeros(N)
RE_rf_mape_x1 = np.zeros(N) # mape_x1
FE_rf_mape_x1 = np.zeros(N)
FEt_rf_mape_x1 = np.zeros(N)
RE_rf_mse_x2 = np.zeros(N) # mse_x2
FE_rf_mse_x2 = np.zeros(N)
FEt_rf_mse_x2 = np.zeros(N)
RE_rf_mae_x2 = np.zeros(N) # mae_x2
FE_rf_mae_x2 = np.zeros(N)
FEt_rf_mae_x2 = np.zeros(N) 
RE_rf_mpe_x2 = np.zeros(N) # mpe_x2
FE_rf_mpe_x2 = np.zeros(N)
FEt_rf_mpe_x2 = np.zeros(N)
RE_rf_mape_x2 = np.zeros(N) # mape_x2
FE_rf_mape_x2 = np.zeros(N)
FEt_rf_mape_x2 = np.zeros(N)
RE_rf_mse_x3 = np.zeros(N) # mse_x3
FE_rf_mse_x3 = np.zeros(N)
FEt_rf_mse_x3 = np.zeros(N)
RE_rf_mae_x3 = np.zeros(N) # mae_x2
FE_rf_mae_x3 = np.zeros(N)
FEt_rf_mae_x3 = np.zeros(N)
RE_rf_mpe_x3 = np.zeros(N) # mpe_x2
FE_rf_mpe_x3 = np.zeros(N)
FEt_rf_mpe_x3 = np.zeros(N)
RE_rf_mape_x3 = np.zeros(N) # mape_x2
FE_rf_mape_x3 = np.zeros(N)
FEt_rf_mape_x3 = np.zeros(N)
CIlo_RE_rf_b0 = np.zeros(N) # CI_b0
CIlo_FE_rf_b0 = np.zeros(N)
CIlo_FEt_rf_b0 = np.zeros(N)
CIhi_RE_rf_b0 = np.zeros(N)
CIhi_FE_rf_b0 = np.zeros(N)
CIhi_FEt_rf_b0 = np.zeros(N)
CIlo_RE_rf_b1 = np.zeros(N) # CI_b1
CIlo_FE_rf_b1 = np.zeros(N)
CIlo_FEt_rf_b1 = np.zeros(N)
CIhi_RE_rf_b1 = np.zeros(N)
CIhi_FE_rf_b1 = np.zeros(N)
CIhi_FEt_rf_b1 = np.zeros(N)
CIlo_RE_rf_b2 = np.zeros(N) # CI_b2
CIlo_FE_rf_b2 = np.zeros(N)
CIlo_FEt_rf_b2 = np.zeros(N)
CIhi_RE_rf_b2 = np.zeros(N)
CIhi_FE_rf_b2 = np.zeros(N)
CIhi_FEt_rf_b2 = np.zeros(N)
CIlo_RE_rf_b3 = np.zeros(N) # CI_b3
CIlo_FE_rf_b3 = np.zeros(N)
CIlo_FEt_rf_b3 = np.zeros(N)
CIhi_RE_rf_b3 = np.zeros(N)
CIhi_FE_rf_b3 = np.zeros(N)
CIhi_FEt_rf_b3 = np.zeros(N)

# lgb
RE_lgb_b0Var = np.zeros(N) # varb0
FE_lgb_b0Var = np.zeros(N)
FEt_lgb_b0Var = np.zeros(N)
RE_lgb_b1Var = np.zeros(N) # varb1
FE_lgb_b1Var = np.zeros(N)
FEt_lgb_b1Var = np.zeros(N)
RE_lgb_b2Var = np.zeros(N) # varb2
FE_lgb_b2Var = np.zeros(N)
FEt_lgb_b2Var = np.zeros(N)
RE_lgb_b3Var = np.zeros(N) # varb3
FE_lgb_b3Var = np.zeros(N)
FEt_lgb_b3Var = np.zeros(N)
RE_lgb_b0 = np.zeros(N) # b0
FE_lgb_b0 = np.zeros(N)
FEt_lgb_b0 = np.zeros(N)
RE_lgb_b1 = np.zeros(N) # b1
FE_lgb_b1 = np.zeros(N)
FEt_lgb_b1 = np.zeros(N)
RE_lgb_b2 = np.zeros(N) # b2
FE_lgb_b2 = np.zeros(N)
FEt_lgb_b2 = np.zeros(N)
RE_lgb_b3 = np.zeros(N) # b3
FE_lgb_b3 = np.zeros(N)
FEt_lgb_b3 = np.zeros(N)
RE_lgb_sig2 = np.zeros(N) # sig2
FE_lgb_sig2 = np.zeros(N)
FEt_lgb_sig2 = np.zeros(N)
RE_lgb_b0SE = np.zeros(N) # SE0
FE_lgb_b0SE = np.zeros(N)
FEt_lgb_b0SE = np.zeros(N)
RE_lgb_b1SE = np.zeros(N) # SE1
FE_lgb_b1SE = np.zeros(N)
FEt_lgb_b1SE = np.zeros(N)
RE_lgb_b2SE = np.zeros(N) # SE2
FE_lgb_b2SE = np.zeros(N)
FEt_lgb_b2SE = np.zeros(N) 
RE_lgb_b3SE = np.zeros(N) # SE3
FE_lgb_b3SE = np.zeros(N)
FEt_lgb_b3SE = np.zeros(N)
RE_lgb_r2 = np.zeros(N) # r2
FE_lgb_r2 = np.zeros(N)
FEt_lgb_r2 = np.zeros(N)
RE_lgb_a_r2 = np.zeros(N) # ar2
FE_lgb_a_r2 = np.zeros(N)
FEt_lgb_a_r2 = np.zeros(N)
RE_lgb_aic = np.zeros(N) # aic
FE_lgb_aic = np.zeros(N)
FEt_lgb_aic = np.zeros(N)
RE_lgb_mse = np.zeros(N) # mse_y
FE_lgb_mse = np.zeros(N)
FEt_lgb_mse = np.zeros(N)
RE_lgb_mae = np.zeros(N) # mae_y
FE_lgb_mae = np.zeros(N)
FEt_lgb_mae = np.zeros(N)
RE_lgb_mpe = np.zeros(N) # mpe_y
FE_lgb_mpe = np.zeros(N)
FEt_lgb_mpe = np.zeros(N)
RE_lgb_mape = np.zeros(N) # mape_y
FE_lgb_mape = np.zeros(N)
FEt_lgb_mape = np.zeros(N)
RE_lgb_mse_x1 = np.zeros(N) # mse_x1
FE_lgb_mse_x1 = np.zeros(N)
FEt_lgb_mse_x1 = np.zeros(N) 
RE_lgb_mae_x1 = np.zeros(N) # mae_x1
FE_lgb_mae_x1 = np.zeros(N)
FEt_lgb_mae_x1 = np.zeros(N)
RE_lgb_mpe_x1 = np.zeros(N) # mpe_x1
FE_lgb_mpe_x1 = np.zeros(N)
FEt_lgb_mpe_x1 = np.zeros(N)
RE_lgb_mape_x1 = np.zeros(N) # mape_x1
FE_lgb_mape_x1 = np.zeros(N)
FEt_lgb_mape_x1 = np.zeros(N)
RE_lgb_mse_x2 = np.zeros(N) # mse_x2
FE_lgb_mse_x2 = np.zeros(N)
FEt_lgb_mse_x2 = np.zeros(N)
RE_lgb_mae_x2 = np.zeros(N) # mae_x2
FE_lgb_mae_x2 = np.zeros(N)
FEt_lgb_mae_x2 = np.zeros(N) 
RE_lgb_mpe_x2 = np.zeros(N) # mpe_x2
FE_lgb_mpe_x2 = np.zeros(N)
FEt_lgb_mpe_x2 = np.zeros(N)
RE_lgb_mape_x2 = np.zeros(N) # mape_x2
FE_lgb_mape_x2 = np.zeros(N)
FEt_lgb_mape_x2 = np.zeros(N)
RE_lgb_mse_x3 = np.zeros(N) # mse_x3
FE_lgb_mse_x3 = np.zeros(N)
FEt_lgb_mse_x3 = np.zeros(N)
RE_lgb_mae_x3 = np.zeros(N) # mae_x2
FE_lgb_mae_x3 = np.zeros(N)
FEt_lgb_mae_x3 = np.zeros(N)
RE_lgb_mpe_x3 = np.zeros(N) # mpe_x2
FE_lgb_mpe_x3 = np.zeros(N)
FEt_lgb_mpe_x3 = np.zeros(N)
RE_lgb_mape_x3 = np.zeros(N) # mape_x2
FE_lgb_mape_x3 = np.zeros(N)
FEt_lgb_mape_x3 = np.zeros(N)
CIlo_RE_lgb_b0 = np.zeros(N) # CI_b0
CIlo_FE_lgb_b0 = np.zeros(N)
CIlo_FEt_lgb_b0 = np.zeros(N)
CIhi_RE_lgb_b0 = np.zeros(N)
CIhi_FE_lgb_b0 = np.zeros(N)
CIhi_FEt_lgb_b0 = np.zeros(N)
CIlo_RE_lgb_b1 = np.zeros(N) # CI_b1
CIlo_FE_lgb_b1 = np.zeros(N)
CIlo_FEt_lgb_b1 = np.zeros(N)
CIhi_RE_lgb_b1 = np.zeros(N)
CIhi_FE_lgb_b1 = np.zeros(N)
CIhi_FEt_lgb_b1 = np.zeros(N)
CIlo_RE_lgb_b2 = np.zeros(N) # CI_b2
CIlo_FE_lgb_b2 = np.zeros(N)
CIlo_FEt_lgb_b2 = np.zeros(N)
CIhi_RE_lgb_b2 = np.zeros(N)
CIhi_FE_lgb_b2 = np.zeros(N)
CIhi_FEt_lgb_b2 = np.zeros(N)
CIlo_RE_lgb_b3 = np.zeros(N) # CI_b3
CIlo_FE_lgb_b3 = np.zeros(N)
CIlo_FEt_lgb_b3 = np.zeros(N)
CIhi_RE_lgb_b3 = np.zeros(N)
CIhi_FE_lgb_b3 = np.zeros(N)
CIhi_FEt_lgb_b3 = np.zeros(N)

# mlp
RE_mlp_b0Var = np.zeros(N) # varb0
FE_mlp_b0Var = np.zeros(N)
FEt_mlp_b0Var = np.zeros(N)
RE_mlp_b1Var = np.zeros(N) # varb1
FE_mlp_b1Var = np.zeros(N)
FEt_mlp_b1Var = np.zeros(N)
RE_mlp_b2Var = np.zeros(N) # varb2
FE_mlp_b2Var = np.zeros(N)
FEt_mlp_b2Var = np.zeros(N)
RE_mlp_b3Var = np.zeros(N) # varb3
FE_mlp_b3Var = np.zeros(N)
FEt_mlp_b3Var = np.zeros(N)
RE_mlp_b0 = np.zeros(N) # b0
FE_mlp_b0 = np.zeros(N)
FEt_mlp_b0 = np.zeros(N)
RE_mlp_b1 = np.zeros(N) # b1
FE_mlp_b1 = np.zeros(N)
FEt_mlp_b1 = np.zeros(N)
RE_mlp_b2 = np.zeros(N) # b2
FE_mlp_b2 = np.zeros(N)
FEt_mlp_b2 = np.zeros(N)
RE_mlp_b3 = np.zeros(N) # b3
FE_mlp_b3 = np.zeros(N)
FEt_mlp_b3 = np.zeros(N)
RE_mlp_sig2 = np.zeros(N) # sig2
FE_mlp_sig2 = np.zeros(N)
FEt_mlp_sig2 = np.zeros(N)
RE_mlp_b0SE = np.zeros(N) # SE0
FE_mlp_b0SE = np.zeros(N)
FEt_mlp_b0SE = np.zeros(N)
RE_mlp_b1SE = np.zeros(N) # SE1
FE_mlp_b1SE = np.zeros(N)
FEt_mlp_b1SE = np.zeros(N)
RE_mlp_b2SE = np.zeros(N) # SE2
FE_mlp_b2SE = np.zeros(N)
FEt_mlp_b2SE = np.zeros(N) 
RE_mlp_b3SE = np.zeros(N) # SE3
FE_mlp_b3SE = np.zeros(N)
FEt_mlp_b3SE = np.zeros(N)
RE_mlp_r2 = np.zeros(N) # r2
FE_mlp_r2 = np.zeros(N)
FEt_mlp_r2 = np.zeros(N)
RE_mlp_a_r2 = np.zeros(N) # ar2
FE_mlp_a_r2 = np.zeros(N)
FEt_mlp_a_r2 = np.zeros(N)
RE_mlp_aic = np.zeros(N) # aic
FE_mlp_aic = np.zeros(N)
FEt_mlp_aic = np.zeros(N)
RE_mlp_mse = np.zeros(N) # mse_y
FE_mlp_mse = np.zeros(N)
FEt_mlp_mse = np.zeros(N)
RE_mlp_mae = np.zeros(N) # mae_y
FE_mlp_mae = np.zeros(N)
FEt_mlp_mae = np.zeros(N)
RE_mlp_mpe = np.zeros(N) # mpe_y
FE_mlp_mpe = np.zeros(N)
FEt_mlp_mpe = np.zeros(N)
RE_mlp_mape = np.zeros(N) # mape_y
FE_mlp_mape = np.zeros(N)
FEt_mlp_mape = np.zeros(N)
RE_mlp_mse_x1 = np.zeros(N) # mse_x1
FE_mlp_mse_x1 = np.zeros(N)
FEt_mlp_mse_x1 = np.zeros(N) 
RE_mlp_mae_x1 = np.zeros(N) # mae_x1
FE_mlp_mae_x1 = np.zeros(N)
FEt_mlp_mae_x1 = np.zeros(N)
RE_mlp_mpe_x1 = np.zeros(N) # mpe_x1
FE_mlp_mpe_x1 = np.zeros(N)
FEt_mlp_mpe_x1 = np.zeros(N)
RE_mlp_mape_x1 = np.zeros(N) # mape_x1
FE_mlp_mape_x1 = np.zeros(N)
FEt_mlp_mape_x1 = np.zeros(N)
RE_mlp_mse_x2 = np.zeros(N) # mse_x2
FE_mlp_mse_x2 = np.zeros(N)
FEt_mlp_mse_x2 = np.zeros(N)
RE_mlp_mae_x2 = np.zeros(N) # mae_x2
FE_mlp_mae_x2 = np.zeros(N)
FEt_mlp_mae_x2 = np.zeros(N) 
RE_mlp_mpe_x2 = np.zeros(N) # mpe_x2
FE_mlp_mpe_x2 = np.zeros(N)
FEt_mlp_mpe_x2 = np.zeros(N)
RE_mlp_mape_x2 = np.zeros(N) # mape_x2
FE_mlp_mape_x2 = np.zeros(N)
FEt_mlp_mape_x2 = np.zeros(N)
RE_mlp_mse_x3 = np.zeros(N) # mse_x3
FE_mlp_mse_x3 = np.zeros(N)
FEt_mlp_mse_x3 = np.zeros(N)
RE_mlp_mae_x3 = np.zeros(N) # mae_x2
FE_mlp_mae_x3 = np.zeros(N)
FEt_mlp_mae_x3 = np.zeros(N)
RE_mlp_mpe_x3 = np.zeros(N) # mpe_x2
FE_mlp_mpe_x3 = np.zeros(N)
FEt_mlp_mpe_x3 = np.zeros(N)
RE_mlp_mape_x3 = np.zeros(N) # mape_x2
FE_mlp_mape_x3 = np.zeros(N)
FEt_mlp_mape_x3 = np.zeros(N)
CIlo_RE_mlp_b0 = np.zeros(N) # CI_b0
CIlo_FE_mlp_b0 = np.zeros(N)
CIlo_FEt_mlp_b0 = np.zeros(N)
CIhi_RE_mlp_b0 = np.zeros(N)
CIhi_FE_mlp_b0 = np.zeros(N)
CIhi_FEt_mlp_b0 = np.zeros(N)
CIlo_RE_mlp_b1 = np.zeros(N) # CI_b1
CIlo_FE_mlp_b1 = np.zeros(N)
CIlo_FEt_mlp_b1 = np.zeros(N)
CIhi_RE_mlp_b1 = np.zeros(N)
CIhi_FE_mlp_b1 = np.zeros(N)
CIhi_FEt_mlp_b1 = np.zeros(N)
CIlo_RE_mlp_b2 = np.zeros(N) # CI_b2
CIlo_FE_mlp_b2 = np.zeros(N)
CIlo_FEt_mlp_b2 = np.zeros(N)
CIhi_RE_mlp_b2 = np.zeros(N)
CIhi_FE_mlp_b2 = np.zeros(N)
CIhi_FEt_mlp_b2 = np.zeros(N)
CIlo_RE_mlp_b3 = np.zeros(N) # CI_b3
CIlo_FE_mlp_b3 = np.zeros(N)
CIlo_FEt_mlp_b3 = np.zeros(N)
CIhi_RE_mlp_b3 = np.zeros(N)
CIhi_FE_mlp_b3 = np.zeros(N)
CIhi_FEt_mlp_b3 = np.zeros(N)

# vae
RE_vae_b0Var = np.zeros(N) # varb0
FE_vae_b0Var = np.zeros(N)
FEt_vae_b0Var = np.zeros(N)
RE_vae_b1Var = np.zeros(N) # varb1
FE_vae_b1Var = np.zeros(N)
FEt_vae_b1Var = np.zeros(N)
RE_vae_b2Var = np.zeros(N) # varb2
FE_vae_b2Var = np.zeros(N)
FEt_vae_b2Var = np.zeros(N)
RE_vae_b3Var = np.zeros(N) # varb3
FE_vae_b3Var = np.zeros(N)
FEt_vae_b3Var = np.zeros(N)
RE_vae_b0 = np.zeros(N) # b0
FE_vae_b0 = np.zeros(N)
FEt_vae_b0 = np.zeros(N)	
RE_vae_b1 = np.zeros(N) # b1
FE_vae_b1 = np.zeros(N)
FEt_vae_b1 = np.zeros(N)
RE_vae_b2 = np.zeros(N) # b2
FE_vae_b2 = np.zeros(N)
FEt_vae_b2 = np.zeros(N)
RE_vae_b3 = np.zeros(N) # b3
FE_vae_b3 = np.zeros(N)
FEt_vae_b3 = np.zeros(N)
RE_vae_sig2 = np.zeros(N) # sig2
FE_vae_sig2 = np.zeros(N)
FEt_vae_sig2 = np.zeros(N)
RE_vae_b0SE = np.zeros(N) # SE0
FE_vae_b0SE = np.zeros(N)
FEt_vae_b0SE = np.zeros(N)
RE_vae_b1SE = np.zeros(N) # SE1
FE_vae_b1SE = np.zeros(N)
FEt_vae_b1SE = np.zeros(N)
RE_vae_b2SE = np.zeros(N) # SE2
FE_vae_b2SE = np.zeros(N)
FEt_vae_b2SE = np.zeros(N) 
RE_vae_b3SE = np.zeros(N) # SE3
FE_vae_b3SE = np.zeros(N)
FEt_vae_b3SE = np.zeros(N)
RE_vae_r2 = np.zeros(N) # r2
FE_vae_r2 = np.zeros(N)
FEt_vae_r2 = np.zeros(N)
RE_vae_a_r2 = np.zeros(N) # ar2
FE_vae_a_r2 = np.zeros(N)
FEt_vae_a_r2 = np.zeros(N)
RE_vae_aic = np.zeros(N) # aic
FE_vae_aic = np.zeros(N)
FEt_vae_aic = np.zeros(N)
RE_vae_mse = np.zeros(N) # mse_y
FE_vae_mse = np.zeros(N)
FEt_vae_mse = np.zeros(N)
RE_vae_mae = np.zeros(N) # mae_y
FE_vae_mae = np.zeros(N)
FEt_vae_mae = np.zeros(N)
RE_vae_mpe = np.zeros(N) # mpe_y
FE_vae_mpe = np.zeros(N)
FEt_vae_mpe = np.zeros(N)
RE_vae_mape = np.zeros(N) # mape_y
FE_vae_mape = np.zeros(N)
FEt_vae_mape = np.zeros(N)
RE_vae_mse_x1 = np.zeros(N) # mse_x1
FE_vae_mse_x1 = np.zeros(N)
FEt_vae_mse_x1 = np.zeros(N) 
RE_vae_mae_x1 = np.zeros(N) # mae_x1
FE_vae_mae_x1 = np.zeros(N)
FEt_vae_mae_x1 = np.zeros(N)
RE_vae_mpe_x1 = np.zeros(N) # mpe_x1
FE_vae_mpe_x1 = np.zeros(N)
FEt_vae_mpe_x1 = np.zeros(N)
RE_vae_mape_x1 = np.zeros(N) # mape_x1
FE_vae_mape_x1 = np.zeros(N)
FEt_vae_mape_x1 = np.zeros(N)
RE_vae_mse_x2 = np.zeros(N) # mse_x2
FE_vae_mse_x2 = np.zeros(N)
FEt_vae_mse_x2 = np.zeros(N)
RE_vae_mae_x2 = np.zeros(N) # mae_x2
FE_vae_mae_x2 = np.zeros(N)
FEt_vae_mae_x2 = np.zeros(N) 
RE_vae_mpe_x2 = np.zeros(N) # mpe_x2
FE_vae_mpe_x2 = np.zeros(N)
FEt_vae_mpe_x2 = np.zeros(N)
RE_vae_mape_x2 = np.zeros(N) # mape_x2
FE_vae_mape_x2 = np.zeros(N)
FEt_vae_mape_x2 = np.zeros(N)
RE_vae_mse_x3 = np.zeros(N) # mse_x3
FE_vae_mse_x3 = np.zeros(N)
FEt_vae_mse_x3 = np.zeros(N)
RE_vae_mae_x3 = np.zeros(N) # mae_x2
FE_vae_mae_x3 = np.zeros(N)
FEt_vae_mae_x3 = np.zeros(N)
RE_vae_mpe_x3 = np.zeros(N) # mpe_x2
FE_vae_mpe_x3 = np.zeros(N)
FEt_vae_mpe_x3 = np.zeros(N)
RE_vae_mape_x3 = np.zeros(N) # mape_x2
FE_vae_mape_x3 = np.zeros(N)
FEt_vae_mape_x3 = np.zeros(N)
CIlo_RE_vae_b0 = np.zeros(N) # CI_b0
CIlo_FE_vae_b0 = np.zeros(N)
CIlo_FEt_vae_b0 = np.zeros(N)
CIhi_RE_vae_b0 = np.zeros(N)
CIhi_FE_vae_b0 = np.zeros(N)
CIhi_FEt_vae_b0 = np.zeros(N)
CIlo_RE_vae_b1 = np.zeros(N) # CI_b1
CIlo_FE_vae_b1 = np.zeros(N)
CIlo_FEt_vae_b1 = np.zeros(N)
CIhi_RE_vae_b1 = np.zeros(N)
CIhi_FE_vae_b1 = np.zeros(N)
CIhi_FEt_vae_b1 = np.zeros(N)
CIlo_RE_vae_b2 = np.zeros(N) # CI_b2
CIlo_FE_vae_b2 = np.zeros(N)
CIlo_FEt_vae_b2 = np.zeros(N)
CIhi_RE_vae_b2 = np.zeros(N)
CIhi_FE_vae_b2 = np.zeros(N)
CIhi_FEt_vae_b2 = np.zeros(N)
CIlo_RE_vae_b3 = np.zeros(N) # CI_b3
CIlo_FE_vae_b3 = np.zeros(N)
CIlo_FEt_vae_b3 = np.zeros(N)
CIhi_RE_vae_b3 = np.zeros(N)
CIhi_FE_vae_b3 = np.zeros(N)
CIhi_FEt_vae_b3 = np.zeros(N)

# gae
RE_gae_b0Var = np.zeros(N) # varb0
FE_gae_b0Var = np.zeros(N)
FEt_gae_b0Var = np.zeros(N)
RE_gae_b1Var = np.zeros(N) # varb1
FE_gae_b1Var = np.zeros(N)
FEt_gae_b1Var = np.zeros(N)
RE_gae_b2Var = np.zeros(N) # varb2
FE_gae_b2Var = np.zeros(N)
FEt_gae_b2Var = np.zeros(N)
RE_gae_b3Var = np.zeros(N) # varb3
FE_gae_b3Var = np.zeros(N)
FEt_gae_b3Var = np.zeros(N)
RE_gae_b0 = np.zeros(N) # b0
FE_gae_b0 = np.zeros(N)	
FEt_gae_b0 = np.zeros(N)	
RE_gae_b1 = np.zeros(N) # b1
FE_gae_b1 = np.zeros(N)
FEt_gae_b1 = np.zeros(N)
RE_gae_b2 = np.zeros(N) # b2
FE_gae_b2 = np.zeros(N)
FEt_gae_b2 = np.zeros(N)
RE_gae_b3 = np.zeros(N) # b3
FE_gae_b3 = np.zeros(N)
FEt_gae_b3 = np.zeros(N)
RE_gae_sig2 = np.zeros(N) # sig2
FE_gae_sig2 = np.zeros(N)
FEt_gae_sig2 = np.zeros(N)
RE_gae_b0SE = np.zeros(N) # SE0
FE_gae_b0SE = np.zeros(N)
FEt_gae_b0SE = np.zeros(N)
RE_gae_b1SE = np.zeros(N) # SE1
FE_gae_b1SE = np.zeros(N)
FEt_gae_b1SE = np.zeros(N)
RE_gae_b2SE = np.zeros(N) # SE2
FE_gae_b2SE = np.zeros(N)
FEt_gae_b2SE = np.zeros(N) 
RE_gae_b3SE = np.zeros(N) # SE3
FE_gae_b3SE = np.zeros(N)
FEt_gae_b3SE = np.zeros(N)
RE_gae_r2 = np.zeros(N) # r2
FE_gae_r2 = np.zeros(N)
FEt_gae_r2 = np.zeros(N)
RE_gae_a_r2 = np.zeros(N) # ar2
FE_gae_a_r2 = np.zeros(N)
FEt_gae_a_r2 = np.zeros(N)
RE_gae_aic = np.zeros(N) # aic
FE_gae_aic = np.zeros(N)
FEt_gae_aic = np.zeros(N)
RE_gae_mse = np.zeros(N) # mse_y
FE_gae_mse = np.zeros(N)
FEt_gae_mse = np.zeros(N)
RE_gae_mae = np.zeros(N) # mae_y
FE_gae_mae = np.zeros(N)
FEt_gae_mae = np.zeros(N)
RE_gae_mpe = np.zeros(N) # mpe_y
FE_gae_mpe = np.zeros(N)
FEt_gae_mpe = np.zeros(N)
RE_gae_mape = np.zeros(N) # mape_y
FE_gae_mape = np.zeros(N)
FEt_gae_mape = np.zeros(N)
RE_gae_mse_x1 = np.zeros(N) # mse_x1
FE_gae_mse_x1 = np.zeros(N)
FEt_gae_mse_x1 = np.zeros(N) 
RE_gae_mae_x1 = np.zeros(N) # mae_x1
FE_gae_mae_x1 = np.zeros(N)
FEt_gae_mae_x1 = np.zeros(N)
RE_gae_mpe_x1 = np.zeros(N) # mpe_x1
FE_gae_mpe_x1 = np.zeros(N)
FEt_gae_mpe_x1 = np.zeros(N)
RE_gae_mape_x1 = np.zeros(N) # mape_x1
FE_gae_mape_x1 = np.zeros(N)
FEt_gae_mape_x1 = np.zeros(N)
RE_gae_mse_x2 = np.zeros(N) # mse_x2
FE_gae_mse_x2 = np.zeros(N)
FEt_gae_mse_x2 = np.zeros(N)
RE_gae_mae_x2 = np.zeros(N) # mae_x2
FE_gae_mae_x2 = np.zeros(N)
FEt_gae_mae_x2 = np.zeros(N) 
RE_gae_mpe_x2 = np.zeros(N) # mpe_x2
FE_gae_mpe_x2 = np.zeros(N)
FEt_gae_mpe_x2 = np.zeros(N)
RE_gae_mape_x2 = np.zeros(N) # mape_x2
FE_gae_mape_x2 = np.zeros(N)
FEt_gae_mape_x2 = np.zeros(N)
RE_gae_mse_x3 = np.zeros(N) # mse_x3
FE_gae_mse_x3 = np.zeros(N)
FEt_gae_mse_x3 = np.zeros(N)
RE_gae_mae_x3 = np.zeros(N) # mae_x2
FE_gae_mae_x3 = np.zeros(N)
FEt_gae_mae_x3 = np.zeros(N)
RE_gae_mpe_x3 = np.zeros(N) # mpe_x2
FE_gae_mpe_x3 = np.zeros(N)
FEt_gae_mpe_x3 = np.zeros(N)
RE_gae_mape_x3 = np.zeros(N) # mape_x2
FE_gae_mape_x3 = np.zeros(N)
FEt_gae_mape_x3 = np.zeros(N)
CIlo_RE_gae_b0 = np.zeros(N) # CI_b0
CIlo_FE_gae_b0 = np.zeros(N)
CIlo_FEt_gae_b0 = np.zeros(N)
CIhi_RE_gae_b0 = np.zeros(N)
CIhi_FE_gae_b0 = np.zeros(N)
CIhi_FEt_gae_b0 = np.zeros(N)
CIlo_RE_gae_b1 = np.zeros(N) # CI_b1
CIlo_FE_gae_b1 = np.zeros(N)
CIlo_FEt_gae_b1 = np.zeros(N)
CIhi_RE_gae_b1 = np.zeros(N)
CIhi_FE_gae_b1 = np.zeros(N)
CIhi_FEt_gae_b1 = np.zeros(N)
CIlo_RE_gae_b2 = np.zeros(N) # CI_b2
CIlo_FE_gae_b2 = np.zeros(N)
CIlo_FEt_gae_b2 = np.zeros(N)
CIhi_RE_gae_b2 = np.zeros(N)
CIhi_FE_gae_b2 = np.zeros(N)
CIhi_FEt_gae_b2 = np.zeros(N)
CIlo_RE_gae_b3 = np.zeros(N) # CI_b3
CIlo_FE_gae_b3 = np.zeros(N)
CIlo_FEt_gae_b3 = np.zeros(N)
CIhi_RE_gae_b3 = np.zeros(N)
CIhi_FE_gae_b3 = np.zeros(N)
CIhi_FEt_gae_b3 = np.zeros(N)

# dif
RE_dif_b0Var = np.zeros(N) # varb0
FE_dif_b0Var = np.zeros(N)
FEt_dif_b0Var = np.zeros(N)
RE_dif_b1Var = np.zeros(N) # varb1
FE_dif_b1Var = np.zeros(N)
FEt_dif_b1Var = np.zeros(N)
RE_dif_b2Var = np.zeros(N) # varb2
FE_dif_b2Var = np.zeros(N)
FEt_dif_b2Var = np.zeros(N)
RE_dif_b3Var = np.zeros(N) # varb3
FE_dif_b3Var = np.zeros(N)
FEt_dif_b3Var = np.zeros(N)
RE_dif_b0 = np.zeros(N) # b0
FE_dif_b0 = np.zeros(N)	
FEt_dif_b0 = np.zeros(N)	
RE_dif_b1 = np.zeros(N) # b1
FE_dif_b1 = np.zeros(N)
FEt_dif_b1 = np.zeros(N)
RE_dif_b2 = np.zeros(N) # b2
FE_dif_b2 = np.zeros(N)
FEt_dif_b2 = np.zeros(N)
RE_dif_b3 = np.zeros(N) # b3
FE_dif_b3 = np.zeros(N)
FEt_dif_b3 = np.zeros(N)
RE_dif_sig2 = np.zeros(N) # sig2
FE_dif_sig2 = np.zeros(N)
FEt_dif_sig2 = np.zeros(N)
RE_dif_b0SE = np.zeros(N) # SE0
FE_dif_b0SE = np.zeros(N)
FEt_dif_b0SE = np.zeros(N)
RE_dif_b1SE = np.zeros(N) # SE1
FE_dif_b1SE = np.zeros(N)
FEt_dif_b1SE = np.zeros(N)
RE_dif_b2SE = np.zeros(N) # SE2
FE_dif_b2SE = np.zeros(N)
FEt_dif_b2SE = np.zeros(N) 
RE_dif_b3SE = np.zeros(N) # SE3
FE_dif_b3SE = np.zeros(N)
FEt_dif_b3SE = np.zeros(N)
RE_dif_r2 = np.zeros(N) # r2
FE_dif_r2 = np.zeros(N)
FEt_dif_r2 = np.zeros(N)
RE_dif_a_r2 = np.zeros(N) # ar2
FE_dif_a_r2 = np.zeros(N)
FEt_dif_a_r2 = np.zeros(N)
RE_dif_aic = np.zeros(N) # aic
FE_dif_aic = np.zeros(N)
FEt_dif_aic = np.zeros(N)
RE_dif_mse = np.zeros(N) # mse_y
FE_dif_mse = np.zeros(N)
FEt_dif_mse = np.zeros(N)
RE_dif_mae = np.zeros(N) # mae_y
FE_dif_mae = np.zeros(N)
FEt_dif_mae = np.zeros(N)
RE_dif_mpe = np.zeros(N) # mpe_y
FE_dif_mpe = np.zeros(N)
FEt_dif_mpe = np.zeros(N)
RE_dif_mape = np.zeros(N) # mape_y
FE_dif_mape = np.zeros(N)
FEt_dif_mape = np.zeros(N)
RE_dif_mse_x1 = np.zeros(N) # mse_x1
FE_dif_mse_x1 = np.zeros(N)
FEt_dif_mse_x1 = np.zeros(N) 
RE_dif_mae_x1 = np.zeros(N) # mae_x1
FE_dif_mae_x1 = np.zeros(N)
FEt_dif_mae_x1 = np.zeros(N)
RE_dif_mpe_x1 = np.zeros(N) # mpe_x1
FE_dif_mpe_x1 = np.zeros(N)
FEt_dif_mpe_x1 = np.zeros(N)
RE_dif_mape_x1 = np.zeros(N) # mape_x1
FE_dif_mape_x1 = np.zeros(N)
FEt_dif_mape_x1 = np.zeros(N)
RE_dif_mse_x2 = np.zeros(N) # mse_x2
FE_dif_mse_x2 = np.zeros(N)
FEt_dif_mse_x2 = np.zeros(N)
RE_dif_mae_x2 = np.zeros(N) # mae_x2
FE_dif_mae_x2 = np.zeros(N)
FEt_dif_mae_x2 = np.zeros(N) 
RE_dif_mpe_x2 = np.zeros(N) # mpe_x2
FE_dif_mpe_x2 = np.zeros(N)
FEt_dif_mpe_x2 = np.zeros(N)
RE_dif_mape_x2 = np.zeros(N) # mape_x2
FE_dif_mape_x2 = np.zeros(N)
FEt_dif_mape_x2 = np.zeros(N)
RE_dif_mse_x3 = np.zeros(N) # mse_x3
FE_dif_mse_x3 = np.zeros(N)
FEt_dif_mse_x3 = np.zeros(N)
RE_dif_mae_x3 = np.zeros(N) # mae_x2
FE_dif_mae_x3 = np.zeros(N)
FEt_dif_mae_x3 = np.zeros(N)
RE_dif_mpe_x3 = np.zeros(N) # mpe_x2
FE_dif_mpe_x3 = np.zeros(N)
FEt_dif_mpe_x3 = np.zeros(N)
RE_dif_mape_x3 = np.zeros(N) # mape_x2
FE_dif_mape_x3 = np.zeros(N)
FEt_dif_mape_x3 = np.zeros(N)
CIlo_RE_dif_b0 = np.zeros(N) # CI_b0
CIlo_FE_dif_b0 = np.zeros(N)
CIlo_FEt_dif_b0 = np.zeros(N)
CIhi_RE_dif_b0 = np.zeros(N)
CIhi_FE_dif_b0 = np.zeros(N)
CIhi_FEt_dif_b0 = np.zeros(N)
CIlo_RE_dif_b1 = np.zeros(N) # CI_b1
CIlo_FE_dif_b1 = np.zeros(N)
CIlo_FEt_dif_b1 = np.zeros(N)
CIhi_RE_dif_b1 = np.zeros(N)
CIhi_FE_dif_b1 = np.zeros(N)
CIhi_FEt_dif_b1 = np.zeros(N)
CIlo_RE_dif_b2 = np.zeros(N) # CI_b2
CIlo_FE_dif_b2 = np.zeros(N)
CIlo_FEt_dif_b2 = np.zeros(N)
CIhi_RE_dif_b2 = np.zeros(N)
CIhi_FE_dif_b2 = np.zeros(N)
CIhi_FEt_dif_b2 = np.zeros(N)
CIlo_RE_dif_b3 = np.zeros(N) # CI_b3
CIlo_FE_dif_b3 = np.zeros(N)
CIlo_FEt_dif_b3 = np.zeros(N)
CIhi_RE_dif_b3 = np.zeros(N)
CIhi_FE_dif_b3 = np.zeros(N)
CIhi_FEt_dif_b3 = np.zeros(N)



#%%###########################################################################
###                                 MONTE CARLO                            ###
##############################################################################

# Import additional required libraries
# For panel data models: from linearmodels import PanelOLS, RandomEffects
# For mixed effects models: import statsmodels.formula.api as smf
# For robust covariance: from statsmodels.stats.sandwich_covariance import cov_hac

# All of these empty vectors will store various results for each model
  # (i.e. REl, FEs, FE_lt, ...)

import time
#import math
#import random
import numpy as np
#import pandas as pd

### Imputation libraries
from statsmodels.imputation.mice import MICEData           # for MI & LH
import statsmodels.api as sm                               # for LH

#
from sklearn.experimental import enable_iterative_imputer  # RF & LGBM
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim

from pythae.models import MIWAE, MIWAEConfig
from pythae.trainers import BaseTrainerConfig, BaseTrainer
from pythae.data.datasets import BaseDataset

import matplotlib.pyplot as plt
#from scipy.stats import nct
from scipy.stats import t

from sklearn.preprocessing import MinMaxScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size_default = 128
eps = 1e-8

# ------------------------------------------------------------------------
# Precompute reusable values
# ------------------------------------------------------------------------
n_cols_X = 3  # x.1, x.2, x.3
column_mask_cache = {}  # for RF column masks

### To fix seed being reset in ML algorithms
rng = np.random.default_rng()

for k in range(N):
    ### To show which iteration you're on
    print("--------------------------------------------------------------")
    print(f"Iteration {k+1} / {N}")
    print("--------------------------------------------------------------")
    
    #################
    ### SETUP     ###
    #################

    # Initialize
    X = pd.DataFrame()
    y = []

    paper_iterator = 1  # since Country 1 includes 1 paper per year
    countryIndex = 1
    
    if(case==1 or case==3 or case==5):
        for i in range(numCountries):
            for j in range(numYears):
                mu_y = -2 + 0.0541*(5*i+j)  # The mean for y increases by 0.5 per country, also increases by _ per year
                mu_0 = np.array([mu_y, mu_x1, mu_x2, mu_x3])  # x remains mean 0
                
                z_data = rng.multivariate_normal(mean=mu_0, cov=SIGMA, size=n)
                #z_data = np.random.multivariate_normal(mean=mu_0, cov=SIGMA, size=n)
                x = z_data[:, 1:4]
                y2 = z_data[:, 0]
                
                X2 = pd.DataFrame({
                    'constant': 1,
                    'x.1': x[:, 0],
                    'x.2': x[:, 1],
                    'x.3': x[:, 2],
                    'year': j + 2020,
                    'country': countryIndex,
                    'paper': paper_iterator,
                    'trend': 0
                })
                
                X = pd.concat([X, X2], ignore_index=True)
                y.append(y2)
                paper_iterator = paper_iterator + 1
            countryIndex = countryIndex + 1
        y = np.concatenate(y)

    if(case==2 or case==4 or case==6):
        for i in range(numCountries):
            for j in range(numYears):
                mu_y = -10 + 0.2705*(5*i+j)  # The mean for y increases by 0.5 per country, also increases by _ per year
                mu_0 = np.array([mu_y, mu_x1, mu_x2, mu_x3])  # x remains mean 0
                
                z_data = rng.multivariate_normal(mean=mu_0, cov=SIGMA, size=n)
                #z_data = np.random.multivariate_normal(mean=mu_0, cov=SIGMA, size=n)
                x = z_data[:, 1:4]
                y2 = z_data[:, 0]
                
                X2 = pd.DataFrame({
                    'constant': 1,
                    'x.1': x[:, 0],
                    'x.2': x[:, 1],
                    'x.3': x[:, 2],
                    'year': j + 2020,
                    'country': countryIndex,
                    'paper': paper_iterator,
                    'trend': 0
                })
                
                X = pd.concat([X, X2], ignore_index=True)
                y.append(y2)
                paper_iterator = paper_iterator + 1
            countryIndex = countryIndex + 1



    #################################
    ### MISSING DATA CONSTRUCTION ###
    #################################  
    if case > 4:
        missingness = 30
    elif case > 2:
        missingness = 15
    elif case <= 2:
        missingness = 5



    ### Random Missingness ###
    # For every n observations, randomly make "missingness"% of them missing
    
    rows = list(range(len(X)))
    blocks = {}
    for row in rows:
        block_id = math.ceil((row + 1) / n)  # Group rows into blocks
        if block_id not in blocks:
            blocks[block_id] = []
        blocks[block_id].append(row)
    
    missing_rows = []
    
    for b in blocks.values():
        k0 = min(missingness, len(b))  # 6.6 rows per 100
        missing_rows.extend(random.sample(b, int(k0)))
    
    X.loc[missing_rows, 'x.1'] = np.nan


    
    ##################
    ### IMPUTATION ###
    ##################  
    
    def run_all_imputers(X, y, verbose=False):
        imputations = {}

        # CCA
        temp = X.copy(); temp['y'] = y
        imputations['CCA'] = temp.dropna()

        # MI (MICE)
        temp = X.copy(); temp['y'] = y
        original_cols = temp.columns.tolist()
        original_dtypes = temp.dtypes.to_dict()
        int_cols = [c for c in temp.columns if pd.api.types.is_integer_dtype(original_dtypes[c])]

        temp.columns = temp.columns.str.replace(r"[^\w]", "_", regex=True).str.strip("_")
        imp = MICEData(temp)
        for _ in range(5): 
            imp.update_all()

        X_mi = imp.data.copy()
        X_mi.columns = original_cols

        X_mi = X_mi.replace([np.inf, -np.inf], np.nan)

        for col in int_cols:
            col_values = X_mi[col].to_numpy()
            col_values = np.where(np.isfinite(col_values), col_values, np.nan)

            if np.isnan(col_values).any():
                median_val = np.nanmedian(col_values)
                if np.isnan(median_val):
                    median_val = 0
                col_values = np.where(np.isnan(col_values), median_val, col_values)

            X_mi[col] = np.round(col_values).astype(original_dtypes[col])

        imputations['MI'] = X_mi


        # LH
        temp = X.copy(); temp['y'] = y
        original_cols = temp.columns.tolist()
        original_dtypes = temp.dtypes.to_dict()
        int_cols = [c for c in temp.columns if pd.api.types.is_integer_dtype(original_dtypes[c])]

        temp.columns = temp.columns.str.replace(r"[^\w]", "_", regex=True).str.strip("_")
        imp = MICEData(temp)

        for col in temp.columns:
            predictors = [c for c in temp.columns if c != col]
            formula = " + ".join(predictors)
            imp.set_imputer(col, model_class=sm.OLS, formula=formula)

        for _ in range(10): 
            imp.update_all()

        X_lh = imp.data.copy()
        X_lh.columns = original_cols

        X_lh = X_lh.replace([np.inf, -np.inf], np.nan)

        for col in int_cols:
            col_values = X_lh[col].to_numpy()
            col_values = np.where(np.isfinite(col_values), col_values, np.nan)

            if np.isnan(col_values).any():
                median_val = np.nanmedian(col_values)
                if np.isnan(median_val):
                    median_val = 0
                col_values = np.where(np.isnan(col_values), median_val, col_values)

            X_lh[col] = np.round(col_values).astype(original_dtypes[col])

        imputations['LH'] = X_lh


        # RF / LGBM optimized
        temp = X.copy(); temp['y'] = y
        original_dtypes = temp.dtypes.to_dict()
        int_cols = [c for c in temp.columns if pd.api.types.is_integer_dtype(original_dtypes[c])]

        rf_imputer = IterativeImputer(
            estimator=RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1),
            max_iter=10, random_state=0
        )

        X_rf = pd.DataFrame(rf_imputer.fit_transform(temp), columns=temp.columns)

        X_rf = X_rf.replace([np.inf, -np.inf], np.nan)

        for col in int_cols:
            col_values = X_rf[col].to_numpy()
            col_values = np.where(np.isfinite(col_values), col_values, np.nan)

            if np.isnan(col_values).any():
                median_val = np.nanmedian(col_values)
                if np.isnan(median_val):
                    median_val = 0
                col_values = np.where(np.isnan(col_values), median_val, col_values)

            X_rf[col] = np.round(col_values).astype(original_dtypes[col])

        imputations['RF'] = X_rf


        lgb_imputer = IterativeImputer(
            estimator=LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=0, n_jobs=-1),
            max_iter=10, random_state=0
        )

        X_lgb = pd.DataFrame(lgb_imputer.fit_transform(temp), columns=temp.columns)

        X_lgb = X_lgb.replace([np.inf, -np.inf], np.nan)

        for col in int_cols:
            col_values = X_lgb[col].to_numpy()
            col_values = np.where(np.isfinite(col_values), col_values, np.nan)

            if np.isnan(col_values).any():
                median_val = np.nanmedian(col_values)
                if np.isnan(median_val):
                    median_val = 0
                col_values = np.where(np.isnan(col_values), median_val, col_values)

            X_lgb[col] = np.round(col_values).astype(original_dtypes[col])

        imputations['LGBM'] = X_lgb


        # ----------------------------------------------------------------
        # MLP Imputer (unchanged except integer safety)
        # ----------------------------------------------------------------
        target_col = "x.1"

        X_full = X.copy(); X_full['y'] = y
        original_dtypes = X_full.dtypes.to_dict()
        int_cols = [c for c in X_full.columns if pd.api.types.is_integer_dtype(original_dtypes[c])]

        train_mask = ~X_full[target_col].isna()

        X_train = X_full.loc[train_mask].drop(columns=[target_col]).values
        y_train = X_full.loc[train_mask, target_col].values.reshape(-1,1)
        X_missing = X_full.loc[~train_mask].drop(columns=[target_col]).values

        x_scaler = StandardScaler()
        X_train = x_scaler.fit_transform(X_train)

        if len(X_missing) > 0:
            X_missing = x_scaler.transform(X_missing)

        y_scaler = StandardScaler()
        y_train = y_scaler.fit_transform(y_train)

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)

        class MLPImputer(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 32),
                    nn.ReLU(),
                    nn.Linear(32, 16),
                    nn.ReLU(),
                    nn.Linear(16, 1)
                )
            def forward(self, x):
                return self.net(x)

        model = MLPImputer(input_dim=X_train.shape[1])

        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        n_epochs = 100
        batch_size = 128

        for epoch in range(n_epochs):
            idx = torch.randperm(X_train.size(0))
            for i in range(0, X_train.size(0), batch_size):
                batch_idx = idx[i:i+batch_size]
                xb, yb = X_train[batch_idx], y_train[batch_idx]

                loss = criterion(model(xb), yb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()

        if len(X_missing) > 0:
            X_missing_t = torch.tensor(X_missing, dtype=torch.float32)

            with torch.no_grad():
                preds_scaled = model(X_missing_t).numpy()

            preds = y_scaler.inverse_transform(preds_scaled)
            X_full.loc[~train_mask, target_col] = preds.flatten()

        X_full = X_full.replace([np.inf, -np.inf], np.nan)

        for col in int_cols:
            col_values = X_full[col].to_numpy()
            col_values = np.where(np.isfinite(col_values), col_values, np.nan)

            if np.isnan(col_values).any():
                median_val = np.nanmedian(col_values)
                if np.isnan(median_val):
                    median_val = 0
                col_values = np.where(np.isnan(col_values), median_val, col_values)

            X_full[col] = np.round(col_values).astype(original_dtypes[col])

        imputations['MLP'] = X_full

        # ----------------------------------------------------------------
        # VAE / MIWAE Imputer
        # ----------------------------------------------------------------
        temp = X.copy()
        temp['y'] = y
        
        # Save original dtypes
        original_dtypes = temp.dtypes.to_dict()
        int_cols = [c for c, t in original_dtypes.items() if pd.api.types.is_integer_dtype(t)]
        
        data = temp.values.astype(np.float32)
        
        missing_mask = np.isnan(data)
        
        col_means = np.nanmean(data, axis=0)
        data_filled = np.where(missing_mask, col_means, data)
        
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_filled).astype(np.float32)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        dataset = BaseDataset(
            data=torch.tensor(data_scaled),
            labels=torch.zeros(len(data_scaled))
        )
        
        input_dim = data_scaled.shape[1]
        
        model_config = MIWAEConfig(
            input_dim=(input_dim,),
            latent_dim=20,
            n_importance_samples=20
        )
        
        model = MIWAE(model_config).to(device)
        
        training_config = BaseTrainerConfig(
            ###num_epochs=10,
            num_epochs=200,
            learning_rate=1e-4,
            per_device_train_batch_size=len(dataset),
            optimizer_cls="Adam"
        )
        
        trainer = BaseTrainer(
            model=model,
            train_dataset=dataset,
            training_config=training_config
        )
        
        trainer.train()
        
        model.eval()
        
        with torch.no_grad():
        
            batch = dataset[:]
            batch = {k: v.to(device) for k, v in batch.items()}
        
            output = model(batch)
        
            recon = output.recon_x
        
            if recon.dim() == 3:
                recon = recon.mean(dim=0)
        
            recon = recon.cpu().numpy()
        
        data_imputed = data.copy()
        data_imputed[missing_mask] = scaler.inverse_transform(recon)[missing_mask]
        
        df_imp = pd.DataFrame(data_imputed, columns=temp.columns)
        
        # Clean infinities globally
        df_imp = df_imp.replace([np.inf, -np.inf], np.nan)
        
        # Restore integer columns safely
        for col in int_cols:
        
            col_values = df_imp[col].to_numpy()
        
            col_values = np.where(np.isfinite(col_values), col_values, np.nan)
        
            if np.isnan(col_values).any():
                median_val = np.nanmedian(col_values)
                if np.isnan(median_val):
                    median_val = 0
                col_values = np.where(np.isnan(col_values), median_val, col_values)
        
            df_imp[col] = np.round(col_values).astype(original_dtypes[col])
        
        imputations['VAE'] = df_imp
        
        # ----------------------------
        # GAIN Imputer (stable)
        # ----------------------------
        start = time.time()
        
        temp = X.copy()
        temp['y'] = y
        
        # Save original dtypes
        original_dtypes = temp.dtypes.to_dict()
        int_cols = [c for c, t in original_dtypes.items() if pd.api.types.is_integer_dtype(t)]
        
        data = temp.values.astype(np.float32)
        
        missing_mask = np.isnan(data)
        mask = (~missing_mask).astype(np.float32)
        
        # Fill missing values with column means for initialization
        col_means = np.nanmean(data, axis=0)
        data_filled = np.where(missing_mask, col_means, data)
        
        # Scale to [0,1] for GAN
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data_filled)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        X_tensor = torch.tensor(data_scaled, dtype=torch.float32, device=device)
        M_tensor = torch.tensor(mask, dtype=torch.float32, device=device)
        
        n, input_dim = X_tensor.shape
        
        hint_rate = 0.9
        alpha = 100
        n_epochs = 200
        lr = 1e-3
        batch_size = min(batch_size_default, n)
        eps = 1e-8  # for numerical stability
        
        hidden_dim = 128
        
        class Generator(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, input_dim),
                    nn.Sigmoid()
                )
            def forward(self, x, m):
                return self.net(torch.cat([x, m], dim=1))
        
        class Discriminator(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, input_dim),
                    nn.Sigmoid()
                )
            def forward(self, x, h):
                return self.net(torch.cat([x, h], dim=1))
        
        G = Generator(input_dim).to(device)
        D = Discriminator(input_dim).to(device)
        
        G_optimizer = optim.Adam(G.parameters(), lr=lr)
        D_optimizer = optim.Adam(D.parameters(), lr=lr)
        
        # Training loop
        for epoch in range(n_epochs):
            idx = torch.randperm(n)
            for i in range(0, n, batch_size):
                batch_idx = idx[i:i+batch_size]
                X_batch = X_tensor[batch_idx]
                M_batch = M_tensor[batch_idx]
        
                Z = torch.rand_like(X_batch)
                X_hat = M_batch * X_batch + (1 - M_batch) * Z
        
                G_sample = G(X_hat, M_batch)
                X_tilde = M_batch * X_batch + (1 - M_batch) * G_sample
        
                B = torch.bernoulli(torch.full(M_batch.shape, hint_rate, device=device))
                H = B * M_batch + 0.5 * (1 - B)
        
                # Train Discriminator
                D_prob = D(X_tilde.detach(), H)
                D_loss = -torch.mean(
                    M_batch * torch.log(D_prob + eps) +
                    (1 - M_batch) * torch.log(1 - D_prob + eps)
                )
                D_optimizer.zero_grad()
                D_loss.backward()
                D_optimizer.step()
        
                # Train Generator
                D_prob = D(X_tilde, H)
                G_loss_adv = -torch.sum((1 - M_batch) * torch.log(D_prob + eps)) / torch.sum(1 - M_batch)
                G_loss_mse = torch.mean((M_batch * X_batch - M_batch * G_sample)**2) / torch.mean(M_batch)
                G_loss = G_loss_adv + alpha * G_loss_mse
        
                G_optimizer.zero_grad()
                G_loss.backward()
                G_optimizer.step()
        
        # Imputation
        G.eval()
        with torch.no_grad():
            Z = torch.rand_like(X_tensor)
            X_hat = M_tensor * X_tensor + (1 - M_tensor) * Z
            G_sample = G(X_hat, M_tensor)
            X_imputed = M_tensor * X_tensor + (1 - M_tensor) * G_sample
        
        # Back to numpy and original scale
        X_imputed = X_imputed.cpu().numpy()
        data_imputed = scaler.inverse_transform(X_imputed)
        
        df_imp = pd.DataFrame(data_imputed, columns=temp.columns)
        
        # Replace inf/-inf and fill NaNs
        df_imp = df_imp.replace([np.inf, -np.inf], np.nan)
        for col in df_imp.columns:
            if pd.api.types.is_numeric_dtype(df_imp[col]):
                median_val = df_imp[col].median(skipna=True)
                if np.isnan(median_val):
                    median_val = 0
                df_imp[col] = df_imp[col].fillna(median_val)
        
        # Restore integer columns safely
        for col in int_cols:
            df_imp[col] = np.round(df_imp[col]).astype(original_dtypes[col])
        
        imputations['GAIN'] = df_imp
        
        print(f"Done GAIN in {time.time()-start:.2f}s")
        
        # ----------------------------
        # Diffusion Imputer (DDPM-style, FE-safe)
        # ----------------------------
        start = time.time()
        
        temp = X.copy()
        temp["y"] = y
        cols = temp.columns.tolist()
        
        original_dtypes = temp.dtypes.to_dict()
        int_cols = [c for c, t in original_dtypes.items() if pd.api.types.is_integer_dtype(t)]
        
        data = temp.values.astype(np.float32)
        
        miss_col = temp.columns.get_loc("x.1")
        
        missing_mask = np.isnan(data[:, miss_col])
        mask = (~missing_mask).astype(np.float32)
        
        col_mean = np.nanmean(data[:, miss_col])
        data[missing_mask, miss_col] = col_mean
        
        scaler_dif = StandardScaler()
        data_scaled = scaler_dif.fit_transform(data)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        X_tensor = torch.tensor(data_scaled, dtype=torch.float32, device=device)
        M_tensor = torch.ones_like(X_tensor)
        M_tensor[:, miss_col] = torch.tensor(mask, dtype=torch.float32, device=device)
        
        n, dim = X_tensor.shape
        T = 100
        beta_start = 1e-4
        beta_end = 0.02
        
        lr = 1e-3
        n_epochs = 500
        batch_size = min(batch_size_default, n)
        
        betas = torch.linspace(beta_start, beta_end, T, device=device)
        alphas = 1. - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        
        
        class TimeEmbedding(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.lin = nn.Linear(1, dim)
            def forward(self, t):
                return torch.relu(self.lin(t.unsqueeze(-1)))
        
        hidden_dim = 128
        
        class DiffusionModel(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.time_embed = TimeEmbedding(16)
                self.net = nn.Sequential(
                    nn.Linear(dim + dim + 16, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, dim)
                )
            def forward(self, x, t, m):
                t_embed = self.time_embed(t)
                return self.net(torch.cat([x, m, t_embed], dim=1))
        
        
        model = DiffusionModel(dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        for epoch in range(n_epochs):
            idx = torch.randperm(n)
            for i in range(0, n, batch_size):
                batch_idx = idx[i:i+batch_size]
                x0 = X_tensor[batch_idx]
                m = M_tensor[batch_idx]
                t = torch.randint(0, T, (len(batch_idx),), device=device)
                alpha_bar = alpha_bars[t].unsqueeze(1)
                noise = torch.randn_like(x0)
                xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise
                xt = m * x0 + (1 - m) * xt
                noise_pred = model(xt, t.float()/T, m)
                loss = torch.sum((1 - m) * (noise - noise_pred)**2) / torch.sum(1 - m)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        model.eval()
        
        with torch.no_grad():
            x = X_tensor.clone()
            x[:, miss_col] = torch.randn_like(x[:, miss_col])
            for t in reversed(range(T)):
                t_tensor = torch.full((n,), t, device=device)
                alpha = alphas[t]
                alpha_bar = alpha_bars[t]
                beta = betas[t]
                noise_pred = model(x, t_tensor.float()/T, M_tensor)
                x = (1 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1 - alpha_bar)) * noise_pred)
                if t > 0:
                    x[:, miss_col] += torch.sqrt(beta) * torch.randn_like(x[:, miss_col])
                x = M_tensor * X_tensor + (1 - M_tensor) * x
        
        X_imputed = x.cpu().numpy()
        data_imputed = scaler_dif.inverse_transform(X_imputed)
        
        df_imp = pd.DataFrame(data_imputed, columns=cols)
        
        # ----------------------------
        # FE-safe adjustments
        # ----------------------------
        df_imp = df_imp.replace([np.inf, -np.inf], np.nan)
        
        for col in int_cols:
            col_values = df_imp[col].to_numpy()
            col_values = np.where(np.isfinite(col_values), col_values, np.nan)
            if np.isnan(col_values).any():
                median_val = np.nanmedian(col_values)
                if np.isnan(median_val):
                    median_val = 0
                col_values = np.where(np.isnan(col_values), median_val, col_values)
            df_imp[col] = np.round(col_values).astype(original_dtypes[col])
        
        imputations['DIF'] = df_imp
        
        print(f"Done Diffusion (FE-safe) in {time.time()-start:.2f}s")
    
    
    
        return imputations
    
    results = run_all_imputers(X, y, verbose=True)

    # Access/store each imputed DataFrame
    X_cca = results['CCA']
    X_mi = results['MI']
    X_lh = results['LH']
    X_rf = results['RF']
    X_lgb = results['LGBM']
    X_mlp = results['MLP']
    X_vae = results['VAE']
    X_gae = results['GAIN']
    X_dif = results['DIF']
    
    ### Fixes the ARC issue for Diffusion's FE
    # 1) Ensure categorical columns are not all missing/constant
    cat_cols = ['paper']  # add more if you have other categorical columns used in formulas
    for col in cat_cols:
        if col in X_dif.columns:
            if X_dif[col].isna().all():  # all missing
                X_dif[col] = 'missing'
            elif X_dif[col].nunique() <= 1:  # constant
                X_dif[col] = X_dif[col].fillna(X_dif[col].iloc[0])
    
    # 2) Ensure numeric columns are finite (do not overwrite actual imputed values)
    num_cols = X_dif.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        col_values = X_dif[col].to_numpy(copy=True)  # force copy so we can write to it
        # only replace infinite values or remaining NaNs with median
        mask = ~np.isfinite(col_values)
        if mask.any():
            median_val = np.nanmedian(col_values)
            if np.isnan(median_val):
                median_val = 0
            col_values[mask] = median_val
            X_dif[col] = col_values
    
    # To fix ARC issue with power curves
    for df in [X_gae, X_dif]:
        # add tiny jitter to numeric columns to prevent perfect collinearity
        num_cols = ['y', 'x.1', 'x.2', 'x.3']  # adjust if more
        for col in num_cols:
            #df[col] = df[col] + np.random.normal(0, 1e-6, size=len(df))
            df[col] = df[col] + rng.normal(0, 1e-6, size=len(df))
    
    ##################
    ### RUN MODELS ###
    ##################
    
    def arc_safe_regression(model_result):
        """
        Replace zero / NaN standard errors to avoid power curve issues on HPC.
        Works for statsmodels OLS/FE models.
        """
        coefs = model_result.params.copy()
        ses = model_result.bse.copy()
        
        # Replace any 0 or NaN SE with tiny number
        ses[~np.isfinite(ses) | (ses == 0)] = 1e-6
        
        # Replace any NaN coefficient with 0
        coefs[~np.isfinite(coefs)] = 0
        
        return coefs, ses
    
    ##### cca #####
    
    ### Study-level Fixed Effects ###
    FE_cca_ = smf.ols('y ~ x_1 + x_2 + x_3 + C(paper)', data=X_cca.rename(columns={'x.1': 'x_1', 'x.2': 'x_2', 'x.3': 'x_3'})).fit()  # fixed model
    FE_cca_result = FE_cca_.summary()

    FE_cca_r2[k] = FE_cca_.rsquared
    FE_cca_a_r2[k] = FE_cca_.rsquared_adj
    FE_cca_mse[k] = np.mean(FE_cca_.resid**2)  # MSE (df adjusted)
    FE_cca_mae[k] = np.mean(np.abs(FE_cca_.resid))  # MAE (df adjusted)
    FE_cca_mpe[k] = np.mean(FE_cca_.resid / X_cca['y'].values)  # MPE (df adjusted)
    FE_cca_mape[k] = np.abs(FE_cca_mpe[k])
    
    FE_cca_b0[k] = FE_cca_.params['Intercept']
    FE_cca_b1[k] = FE_cca_.params['x_1']
    FE_cca_b2[k] = FE_cca_.params['x_2']
    FE_cca_b3[k] = FE_cca_.params['x_3']
    
    FE_cca_mse_x1[k] = np.mean((FE_cca_b1[k] - b1True)**2)  # MSE (df adjusted)
    FE_cca_mae_x1[k] = np.mean(np.abs(FE_cca_b1[k] - b1True))  # MAE (df adjusted)
    FE_cca_mpe_x1[k] = np.mean((b1True - FE_cca_b1[k]) / b1True)  # MPE (df adjusted)
    FE_cca_mape_x1[k] = np.abs(FE_cca_mpe_x1[k])
    FE_cca_mse_x2[k] = np.mean((FE_cca_b2[k] - b2True)**2)  # MSE (df adjusted)
    FE_cca_mae_x2[k] = np.mean(np.abs(FE_cca_b2[k] - b2True))  # MAE (df adjusted)
    FE_cca_mpe_x2[k] = np.mean((b2True - FE_cca_b2[k]) / b2True)  # MPE (df adjusted)
    FE_cca_mape_x2[k] = np.abs(FE_cca_mpe_x2[k])
    FE_cca_mse_x3[k] = np.mean((FE_cca_b3[k] - b3True)**3)  # MSE (df adjusted)
    FE_cca_mae_x3[k] = np.mean(np.abs(FE_cca_b3[k] - b3True))  # MAE (df adjusted)
    FE_cca_mpe_x3[k] = np.mean((b3True - FE_cca_b3[k]) / b3True)  # MPE (df adjusted)
    FE_cca_mape_x3[k] = np.abs(FE_cca_mpe_x3[k])
    
    FE_cca_sig2[k] = FE_cca_.scale
    
    FE_cca_b0SE[k] = FE_cca_.bse['Intercept']
    FE_cca_b1SE[k] = FE_cca_.bse['x_1']
    FE_cca_b2SE[k] = FE_cca_.bse['x_2']
    FE_cca_b3SE[k] = FE_cca_.bse['x_3']
    
    CIlo_FE_cca_b0[k] = FE_cca_b0[k] - (1.96 * FE_cca_b0SE[k])
    CIhi_FE_cca_b0[k] = FE_cca_b0[k] + (1.96 * FE_cca_b0SE[k])
    CIlo_FE_cca_b1[k] = FE_cca_b1[k] - (1.96 * FE_cca_b1SE[k])
    CIhi_FE_cca_b1[k] = FE_cca_b1[k] + (1.96 * FE_cca_b1SE[k])
    CIlo_FE_cca_b2[k] = FE_cca_b2[k] - (1.96 * FE_cca_b2SE[k])
    CIhi_FE_cca_b2[k] = FE_cca_b2[k] + (1.96 * FE_cca_b2SE[k])
    CIlo_FE_cca_b3[k] = FE_cca_b3[k] - (1.96 * FE_cca_b3SE[k])
    CIhi_FE_cca_b3[k] = FE_cca_b3[k] + (1.96 * FE_cca_b3SE[k])
    
    p = len(FE_cca_.params)
    obs = len(FE_cca_.resid)
    FE_cca_aic[k] = 2 * p + obs * (np.log(2 * np.pi) + FE_cca_.scale)
    
    FE_cca_b1Var[k] = (FE_cca_b1SE[k])**2
    FE_cca_b2Var[k] = (FE_cca_b2SE[k])**2
    FE_cca_b3Var[k] = (FE_cca_b3SE[k])**2
    
    ### Time-level Fixed Effects ###
    FEt_cca_ = smf.ols('y ~ x_1 + x_2 + x_3 + C(year)', data=X_cca.rename(columns={'x.1': 'x_1', 'x.2': 'x_2', 'x.3': 'x_3'})).fit()  # fixed model
    FEt_cca_result = FEt_cca_.summary()
    
    FEt_cca_r2[k] = FEt_cca_.rsquared
    FEt_cca_a_r2[k] = FEt_cca_.rsquared_adj
    FEt_cca_mse[k] = np.mean(FEt_cca_.resid**2)  # MSE (df adjusted)
    FEt_cca_mae[k] = np.mean(np.abs(FEt_cca_.resid))  # MAE (df adjusted)
    FEt_cca_mpe[k] = np.mean(FEt_cca_.resid / X_cca['y'].values)  # MPE (df adjusted)
    FEt_cca_mape[k] = np.abs(FEt_cca_mpe[k])
    
    FEt_cca_b0[k] = FEt_cca_.params['Intercept']
    FEt_cca_b1[k] = FEt_cca_.params['x_1']
    FEt_cca_b2[k] = FEt_cca_.params['x_2']
    FEt_cca_b3[k] = FEt_cca_.params['x_3']
    
    FEt_cca_mse_x1[k] = np.mean((FEt_cca_b1[k] - b1True)**2)  # MSE (df adjusted)
    FEt_cca_mae_x1[k] = np.mean(np.abs(FEt_cca_b1[k] - b1True))  # MAE (df adjusted)
    FEt_cca_mpe_x1[k] = np.mean((b1True - FEt_cca_b1[k]) / b1True)  # MPE (df adjusted)
    FEt_cca_mape_x1[k] = np.abs(FEt_cca_mpe_x1[k])
    FEt_cca_mse_x2[k] = np.mean((FEt_cca_b2[k] - b2True)**2)  # MSE (df adjusted)
    FEt_cca_mae_x2[k] = np.mean(np.abs(FEt_cca_b2[k] - b2True))  # MAE (df adjusted)
    FEt_cca_mpe_x2[k] = np.mean((b2True - FEt_cca_b2[k]) / b2True)  # MPE (df adjusted)
    FEt_cca_mape_x2[k] = np.abs(FEt_cca_mpe_x2[k])
    FEt_cca_mse_x3[k] = np.mean((FEt_cca_b3[k] - b3True)**3)  # MSE (df adjusted)
    FEt_cca_mae_x3[k] = np.mean(np.abs(FEt_cca_b3[k] - b3True))  # MAE (df adjusted)
    FEt_cca_mpe_x3[k] = np.mean((b3True - FEt_cca_b3[k]) / b3True)  # MPE (df adjusted)
    FEt_cca_mape_x3[k] = np.abs(FEt_cca_mpe_x3[k])
    
    FEt_cca_sig2[k] = FEt_cca_.scale
    
    FEt_cca_b0SE[k] = FEt_cca_.bse['Intercept']
    FEt_cca_b1SE[k] = FEt_cca_.bse['x_1']
    FEt_cca_b2SE[k] = FEt_cca_.bse['x_2']
    FEt_cca_b3SE[k] = FEt_cca_.bse['x_3']
    
    CIlo_FEt_cca_b0[k] = FEt_cca_b0[k] - (1.96 * FEt_cca_b0SE[k])
    CIhi_FEt_cca_b0[k] = FEt_cca_b0[k] + (1.96 * FEt_cca_b0SE[k])
    CIlo_FEt_cca_b1[k] = FEt_cca_b1[k] - (1.96 * FEt_cca_b1SE[k])
    CIhi_FEt_cca_b1[k] = FEt_cca_b1[k] + (1.96 * FEt_cca_b1SE[k])
    CIlo_FEt_cca_b2[k] = FEt_cca_b2[k] - (1.96 * FEt_cca_b2SE[k])
    CIhi_FEt_cca_b2[k] = FEt_cca_b2[k] + (1.96 * FEt_cca_b2SE[k])
    CIlo_FEt_cca_b3[k] = FEt_cca_b3[k] - (1.96 * FEt_cca_b3SE[k])
    CIhi_FEt_cca_b3[k] = FEt_cca_b3[k] + (1.96 * FEt_cca_b3SE[k])
    
    p = len(FEt_cca_.params)
    obs = len(FEt_cca_.resid)
    FEt_cca_aic[k] = 2 * p + obs * (np.log(2 * np.pi) + FEt_cca_.scale)

    FEt_cca_b1Var[k] = (FEt_cca_b1SE[k])**2
    FEt_cca_b2Var[k] = (FEt_cca_b2SE[k])**2
    FEt_cca_b3Var[k] = (FEt_cca_b3SE[k])**2

    ### Random Effects (Time Level) ###
    # Prepare data for panel regression
    X_cca = X_cca.rename(columns={'x.1': 'x_1', 'x.2': 'x_2', 'x.3': 'x_3'})
    #X_cca = X_cca.assign(y=y)
    #X_cca = X_cca.set_index('paper')
    #X_cca['y'] = y
    
    #X_cca['t'] = X_cca.groupby('year').cumcount()
    #X_cca = X_cca.set_index(['country', 't'])
    #X_cca = X_cca.set_index(['country', 'year'])
    X_cca = X_cca.set_index(['paper','year'])      #time level
    
    # Fit mixed linear model with random intercept grouped by paper
    RE_cca_ = RandomEffects.from_formula('y ~ 1 + x_1 + x_2 + x_3',data=X_cca).fit()
    
    # Calculate (adjusted) R-squared
    RE_cca_r2[k] = RE_cca_.rsquared
    RE_cca_a_r2[k] = 1 - (1 - RE_cca_.rsquared) * (n - 1) / (n - p - 1)
    
    RE_cca_mse[k] = np.mean(RE_cca_.resids.values**2)  # MSE (df adjusted)
    RE_cca_mae[k] = np.mean(np.abs(RE_cca_.resids.values))  # MAE (df adjusted)
    RE_cca_mpe[k] = np.mean(RE_cca_.resids.values / X_cca['y'].values)  # MPE (df adjusted)
    RE_cca_mape[k] = np.abs(RE_cca_mpe[k])
    
    RE_cca_b0[k] = RE_cca_.params['Intercept']
    RE_cca_b1[k] = RE_cca_.params['x_1']
    RE_cca_b2[k] = RE_cca_.params['x_2']
    RE_cca_b3[k] = RE_cca_.params['x_3']
    
    RE_cca_mse_x1[k] = np.mean((RE_cca_b1[k] - b1True)**2)  # MSE (df adjusted)
    RE_cca_mae_x1[k] = np.mean(np.abs(RE_cca_b1[k] - b1True))  # MAE (df adjusted)
    RE_cca_mpe_x1[k] = np.mean((b1True - RE_cca_b1[k]) / b1True)  # MPE (df adjusted)
    RE_cca_mape_x1[k] = np.abs(RE_cca_mpe_x1[k])
    RE_cca_mse_x2[k] = np.mean((RE_cca_b2[k] - b2True)**2)  # MSE (df adjusted)
    RE_cca_mae_x2[k] = np.mean(np.abs(RE_cca_b2[k] - b2True))  # MAE (df adjusted)
    RE_cca_mpe_x2[k] = np.mean((b2True - RE_cca_b2[k]) / b2True)  # MPE (df adjusted)
    RE_cca_mape_x2[k] = np.abs(RE_cca_mpe_x2[k])
    RE_cca_mse_x3[k] = np.mean((RE_cca_b3[k] - b3True)**2)  # MSE (df adjusted)
    RE_cca_mae_x3[k] = np.mean(np.abs(RE_cca_b3[k] - b3True))  # MAE (df adjusted)
    RE_cca_mpe_x3[k] = np.mean((b3True - RE_cca_b3[k]) / b3True)  # MPE (df adjusted)
    RE_cca_mape_x3[k] = np.abs(RE_cca_mpe_x3[k])
    
    RE_cca_sig2[k] = RE_cca_.resids.var()  # within variance, like FE residual variance
    
    RE_cca_b0SE[k] = RE_cca_.std_errors['Intercept']
    RE_cca_b1SE[k] = RE_cca_.std_errors['x_1']
    RE_cca_b2SE[k] = RE_cca_.std_errors['x_2']
    RE_cca_b3SE[k] = RE_cca_.std_errors['x_3']
    
    CIlo_RE_cca_b0[k] = RE_cca_b0[k] - (1.96 * RE_cca_b0SE[k])
    CIhi_RE_cca_b0[k] = RE_cca_b0[k] + (1.96 * RE_cca_b0SE[k])
    CIlo_RE_cca_b1[k] = RE_cca_b1[k] - (1.96 * RE_cca_b1SE[k])
    CIhi_RE_cca_b1[k] = RE_cca_b1[k] + (1.96 * RE_cca_b1SE[k])
    CIlo_RE_cca_b2[k] = RE_cca_b2[k] - (1.96 * RE_cca_b2SE[k])
    CIhi_RE_cca_b2[k] = RE_cca_b2[k] + (1.96 * RE_cca_b2SE[k])
    CIlo_RE_cca_b3[k] = RE_cca_b3[k] - (1.96 * RE_cca_b3SE[k])
    CIhi_RE_cca_b3[k] = RE_cca_b3[k] + (1.96 * RE_cca_b3SE[k])
    
    p = len(RE_cca_.params)
    obs = RE_cca_.nobs
    sigma2 = RE_cca_.resids.var()

    RE_cca_aic[k] = 2 * p + obs * (np.log(2 * np.pi) + np.log(sigma2) + 1)
    
    RE_cca_b1Var[k] = (RE_cca_b1SE[k])**2
    RE_cca_b2Var[k] = (RE_cca_b2SE[k])**2
    RE_cca_b3Var[k] = (RE_cca_b3SE[k])**2

    ##### mi #####
    
    ### Study-level Fixed Effects ###
    FE_mi_ = smf.ols('y ~ x_1 + x_2 + x_3 + C(paper)', data=X_mi.rename(columns={'x.1': 'x_1', 'x.2': 'x_2', 'x.3': 'x_3'})).fit()  # fixed model
    FE_mi_result = FE_mi_.summary()

    FE_mi_r2[k] = FE_mi_.rsquared
    FE_mi_a_r2[k] = FE_mi_.rsquared_adj
    FE_mi_mse[k] = np.mean(FE_mi_.resid**2)  # MSE (df adjusted)
    FE_mi_mae[k] = np.mean(np.abs(FE_mi_.resid))  # MAE (df adjusted)
    FE_mi_mpe[k] = np.mean(FE_mi_.resid / X_mi['y'].values)  # MPE (df adjusted)
    FE_mi_mape[k] = np.abs(FE_mi_mpe[k])
    
    FE_mi_b0[k] = FE_mi_.params['Intercept']
    FE_mi_b1[k] = FE_mi_.params['x_1']
    FE_mi_b2[k] = FE_mi_.params['x_2']
    FE_mi_b3[k] = FE_mi_.params['x_3']
    
    FE_mi_mse_x1[k] = np.mean((FE_mi_b1[k] - b1True)**2)  # MSE (df adjusted)
    FE_mi_mae_x1[k] = np.mean(np.abs(FE_mi_b1[k] - b1True))  # MAE (df adjusted)
    FE_mi_mpe_x1[k] = np.mean((b1True - FE_mi_b1[k]) / b1True)  # MPE (df adjusted)
    FE_mi_mape_x1[k] = np.abs(FE_mi_mpe_x1[k])
    FE_mi_mse_x2[k] = np.mean((FE_mi_b2[k] - b2True)**2)  # MSE (df adjusted)
    FE_mi_mae_x2[k] = np.mean(np.abs(FE_mi_b2[k] - b2True))  # MAE (df adjusted)
    FE_mi_mpe_x2[k] = np.mean((b2True - FE_mi_b2[k]) / b2True)  # MPE (df adjusted)
    FE_mi_mape_x2[k] = np.abs(FE_mi_mpe_x2[k])
    FE_mi_mse_x3[k] = np.mean((FE_mi_b3[k] - b3True)**3)  # MSE (df adjusted)
    FE_mi_mae_x3[k] = np.mean(np.abs(FE_mi_b3[k] - b3True))  # MAE (df adjusted)
    FE_mi_mpe_x3[k] = np.mean((b3True - FE_mi_b3[k]) / b3True)  # MPE (df adjusted)
    FE_mi_mape_x3[k] = np.abs(FE_mi_mpe_x3[k])
    
    FE_mi_sig2[k] = FE_mi_.scale
    
    FE_mi_b0SE[k] = FE_mi_.bse['Intercept']
    FE_mi_b1SE[k] = FE_mi_.bse['x_1']
    FE_mi_b2SE[k] = FE_mi_.bse['x_2']
    FE_mi_b3SE[k] = FE_mi_.bse['x_3']
    
    CIlo_FE_mi_b0[k] = FE_mi_b0[k] - (1.96 * FE_mi_b0SE[k])
    CIhi_FE_mi_b0[k] = FE_mi_b0[k] + (1.96 * FE_mi_b0SE[k])
    CIlo_FE_mi_b1[k] = FE_mi_b1[k] - (1.96 * FE_mi_b1SE[k])
    CIhi_FE_mi_b1[k] = FE_mi_b1[k] + (1.96 * FE_mi_b1SE[k])
    CIlo_FE_mi_b2[k] = FE_mi_b2[k] - (1.96 * FE_mi_b2SE[k])
    CIhi_FE_mi_b2[k] = FE_mi_b2[k] + (1.96 * FE_mi_b2SE[k])
    CIlo_FE_mi_b3[k] = FE_mi_b3[k] - (1.96 * FE_mi_b3SE[k])
    CIhi_FE_mi_b3[k] = FE_mi_b3[k] + (1.96 * FE_mi_b3SE[k])
    
    p = len(FE_mi_.params)
    obs = len(FE_mi_.resid)
    FE_mi_aic[k] = 2 * p + obs * (np.log(2 * np.pi) + FE_mi_.scale)
    
    FE_mi_b1Var[k] = (FE_mi_b1SE[k])**2
    FE_mi_b2Var[k] = (FE_mi_b2SE[k])**2
    FE_mi_b3Var[k] = (FE_mi_b3SE[k])**2
     
    ### Time-level Fixed Effects ###
    FEt_mi_ = smf.ols('y ~ x_1 + x_2 + x_3 + C(year)', data=X_mi.rename(columns={'x.1': 'x_1', 'x.2': 'x_2', 'x.3': 'x_3'})).fit()  # fixed model
    FEt_mi_result = FEt_mi_.summary()
    
    FEt_mi_r2[k] = FEt_mi_.rsquared
    FEt_mi_a_r2[k] = FEt_mi_.rsquared_adj
    FEt_mi_mse[k] = np.mean(FEt_mi_.resid**2)  # MSE (df adjusted)
    FEt_mi_mae[k] = np.mean(np.abs(FEt_mi_.resid))  # MAE (df adjusted)
    FEt_mi_mpe[k] = np.mean(FEt_mi_.resid / X_mi['y'].values)  # MPE (df adjusted)
    FEt_mi_mape[k] = np.abs(FEt_mi_mpe[k])
    
    FEt_mi_b0[k] = FEt_mi_.params['Intercept']
    FEt_mi_b1[k] = FEt_mi_.params['x_1']
    FEt_mi_b2[k] = FEt_mi_.params['x_2']
    FEt_mi_b3[k] = FEt_mi_.params['x_3']
    
    FEt_mi_mse_x1[k] = np.mean((FEt_mi_b1[k] - b1True)**2)  # MSE (df adjusted)
    FEt_mi_mae_x1[k] = np.mean(np.abs(FEt_mi_b1[k] - b1True))  # MAE (df adjusted)
    FEt_mi_mpe_x1[k] = np.mean((b1True - FEt_mi_b1[k]) / b1True)  # MPE (df adjusted)
    FEt_mi_mape_x1[k] = np.abs(FEt_mi_mpe_x1[k])
    FEt_mi_mse_x2[k] = np.mean((FEt_mi_b2[k] - b2True)**2)  # MSE (df adjusted)
    FEt_mi_mae_x2[k] = np.mean(np.abs(FEt_mi_b2[k] - b2True))  # MAE (df adjusted)
    FEt_mi_mpe_x2[k] = np.mean((b2True - FEt_mi_b2[k]) / b2True)  # MPE (df adjusted)
    FEt_mi_mape_x2[k] = np.abs(FEt_mi_mpe_x2[k])
    FEt_mi_mse_x3[k] = np.mean((FEt_mi_b3[k] - b3True)**3)  # MSE (df adjusted)
    FEt_mi_mae_x3[k] = np.mean(np.abs(FEt_mi_b3[k] - b3True))  # MAE (df adjusted)
    FEt_mi_mpe_x3[k] = np.mean((b3True - FEt_mi_b3[k]) / b3True)  # MPE (df adjusted)
    FEt_mi_mape_x3[k] = np.abs(FEt_mi_mpe_x3[k])
    
    FEt_mi_sig2[k] = FEt_mi_.scale
    
    FEt_mi_b0SE[k] = FEt_mi_.bse['Intercept']
    FEt_mi_b1SE[k] = FEt_mi_.bse['x_1']
    FEt_mi_b2SE[k] = FEt_mi_.bse['x_2']
    FEt_mi_b3SE[k] = FEt_mi_.bse['x_3']
    
    CIlo_FEt_mi_b0[k] = FEt_mi_b0[k] - (1.96 * FEt_mi_b0SE[k])
    CIhi_FEt_mi_b0[k] = FEt_mi_b0[k] + (1.96 * FEt_mi_b0SE[k])
    CIlo_FEt_mi_b1[k] = FEt_mi_b1[k] - (1.96 * FEt_mi_b1SE[k])
    CIhi_FEt_mi_b1[k] = FEt_mi_b1[k] + (1.96 * FEt_mi_b1SE[k])
    CIlo_FEt_mi_b2[k] = FEt_mi_b2[k] - (1.96 * FEt_mi_b2SE[k])
    CIhi_FEt_mi_b2[k] = FEt_mi_b2[k] + (1.96 * FEt_mi_b2SE[k])
    CIlo_FEt_mi_b3[k] = FEt_mi_b3[k] - (1.96 * FEt_mi_b3SE[k])
    CIhi_FEt_mi_b3[k] = FEt_mi_b3[k] + (1.96 * FEt_mi_b3SE[k])
    
    p = len(FEt_mi_.params)
    obs = len(FEt_mi_.resid)
    FEt_mi_aic[k] = 2 * p + obs * (np.log(2 * np.pi) + FEt_mi_.scale)

    FEt_mi_b1Var[k] = (FEt_mi_b1SE[k])**2
    FEt_mi_b2Var[k] = (FEt_mi_b2SE[k])**2
    FEt_mi_b3Var[k] = (FEt_mi_b3SE[k])**2

    ### Random Effects (Time Level) ###
    # Prepare data for panel regression
    X_mi = X_mi.rename(columns={'x.1': 'x_1', 'x.2': 'x_2', 'x.3': 'x_3'})
    #X_mi = X_mi.assign(y=y)
    #X_mi = X_mi.set_index('paper')
    #X_mi['y'] = y
    
    #X_mi['t'] = X_mi.groupby('year').cumcount()
    #X_mi = X_mi.set_index(['country', 't'])
    #X_mi = X_mi.set_index(['country', 'year'])
    X_mi = X_mi.set_index(['paper','year'])      #time level 
    
    # Fit mixed linear model with random intercept grouped by paper
    RE_mi_ = RandomEffects.from_formula('y ~ 1 + x_1 + x_2 + x_3',data=X_mi).fit()
    
    # Calculate (adjusted) R-squared
    RE_mi_r2[k] = RE_mi_.rsquared
    RE_mi_a_r2[k] = 1 - (1 - RE_mi_.rsquared) * (n - 1) / (n - p - 1)
    
    RE_mi_mse[k] = np.mean(RE_mi_.resids.values**2)  # MSE (df adjusted)
    RE_mi_mae[k] = np.mean(np.abs(RE_mi_.resids.values))  # MAE (df adjusted)
    RE_mi_mpe[k] = np.mean(RE_mi_.resids.values / X_mi['y'].values)  # MPE (df adjusted)
    RE_mi_mape[k] = np.abs(RE_mi_mpe[k])
    
    RE_mi_b0[k] = RE_mi_.params['Intercept']
    RE_mi_b1[k] = RE_mi_.params['x_1']
    RE_mi_b2[k] = RE_mi_.params['x_2']
    RE_mi_b3[k] = RE_mi_.params['x_3']
    
    RE_mi_mse_x1[k] = np.mean((RE_mi_b1[k] - b1True)**2)  # MSE (df adjusted)
    RE_mi_mae_x1[k] = np.mean(np.abs(RE_mi_b1[k] - b1True))  # MAE (df adjusted)
    RE_mi_mpe_x1[k] = np.mean((b1True - RE_mi_b1[k]) / b1True)  # MPE (df adjusted)
    RE_mi_mape_x1[k] = np.abs(RE_mi_mpe_x1[k])
    RE_mi_mse_x2[k] = np.mean((RE_mi_b2[k] - b2True)**2)  # MSE (df adjusted)
    RE_mi_mae_x2[k] = np.mean(np.abs(RE_mi_b2[k] - b2True))  # MAE (df adjusted)
    RE_mi_mpe_x2[k] = np.mean((b2True - RE_mi_b2[k]) / b2True)  # MPE (df adjusted)
    RE_mi_mape_x2[k] = np.abs(RE_mi_mpe_x2[k])
    RE_mi_mse_x3[k] = np.mean((RE_mi_b3[k] - b3True)**2)  # MSE (df adjusted)
    RE_mi_mae_x3[k] = np.mean(np.abs(RE_mi_b3[k] - b3True))  # MAE (df adjusted)
    RE_mi_mpe_x3[k] = np.mean((b3True - RE_mi_b3[k]) / b3True)  # MPE (df adjusted)
    RE_mi_mape_x3[k] = np.abs(RE_mi_mpe_x3[k])
    
    RE_mi_sig2[k] = RE_mi_.resids.var()  # within variance, like FE residual variance
    
    RE_mi_b0SE[k] = RE_mi_.std_errors['Intercept']
    RE_mi_b1SE[k] = RE_mi_.std_errors['x_1']
    RE_mi_b2SE[k] = RE_mi_.std_errors['x_2']
    RE_mi_b3SE[k] = RE_mi_.std_errors['x_3']
    
    CIlo_RE_mi_b0[k] = RE_mi_b0[k] - (1.96 * RE_mi_b0SE[k])
    CIhi_RE_mi_b0[k] = RE_mi_b0[k] + (1.96 * RE_mi_b0SE[k])
    CIlo_RE_mi_b1[k] = RE_mi_b1[k] - (1.96 * RE_mi_b1SE[k])
    CIhi_RE_mi_b1[k] = RE_mi_b1[k] + (1.96 * RE_mi_b1SE[k])
    CIlo_RE_mi_b2[k] = RE_mi_b2[k] - (1.96 * RE_mi_b2SE[k])
    CIhi_RE_mi_b2[k] = RE_mi_b2[k] + (1.96 * RE_mi_b2SE[k])
    CIlo_RE_mi_b3[k] = RE_mi_b3[k] - (1.96 * RE_mi_b3SE[k])
    CIhi_RE_mi_b3[k] = RE_mi_b3[k] + (1.96 * RE_mi_b3SE[k])
    
    p = len(RE_mi_.params)
    obs = RE_mi_.nobs
    sigma2 = RE_mi_.resids.var()

    RE_mi_aic[k] = 2 * p + obs * (np.log(2 * np.pi) + np.log(sigma2) + 1)

    RE_mi_b1Var[k] = (RE_mi_b1SE[k])**2
    RE_mi_b2Var[k] = (RE_mi_b2SE[k])**2
    RE_mi_b3Var[k] = (RE_mi_b3SE[k])**2



    ##### lh #####
    
    ### Study-level Fixed Effects ###
    FE_lh_ = smf.ols('y ~ x_1 + x_2 + x_3 + C(paper)', data=X_lh.rename(columns={'x.1': 'x_1', 'x.2': 'x_2', 'x.3': 'x_3'})).fit()  # fixed model
    FE_lh_result = FE_lh_.summary()

    FE_lh_r2[k] = FE_lh_.rsquared
    FE_lh_a_r2[k] = FE_lh_.rsquared_adj
    FE_lh_mse[k] = np.mean(FE_lh_.resid**2)  # MSE (df adjusted)
    FE_lh_mae[k] = np.mean(np.abs(FE_lh_.resid))  # MAE (df adjusted)
    FE_lh_mpe[k] = np.mean(FE_lh_.resid / X_lh['y'].values)  # MPE (df adjusted)
    FE_lh_mape[k] = np.abs(FE_lh_mpe[k])
    
    FE_lh_b0[k] = FE_lh_.params['Intercept']
    FE_lh_b1[k] = FE_lh_.params['x_1']
    FE_lh_b2[k] = FE_lh_.params['x_2']
    FE_lh_b3[k] = FE_lh_.params['x_3']
    
    FE_lh_mse_x1[k] = np.mean((FE_lh_b1[k] - b1True)**2)  # MSE (df adjusted)
    FE_lh_mae_x1[k] = np.mean(np.abs(FE_lh_b1[k] - b1True))  # MAE (df adjusted)
    FE_lh_mpe_x1[k] = np.mean((b1True - FE_lh_b1[k]) / b1True)  # MPE (df adjusted)
    FE_lh_mape_x1[k] = np.abs(FE_lh_mpe_x1[k])
    FE_lh_mse_x2[k] = np.mean((FE_lh_b2[k] - b2True)**2)  # MSE (df adjusted)
    FE_lh_mae_x2[k] = np.mean(np.abs(FE_lh_b2[k] - b2True))  # MAE (df adjusted)
    FE_lh_mpe_x2[k] = np.mean((b2True - FE_lh_b2[k]) / b2True)  # MPE (df adjusted)
    FE_lh_mape_x2[k] = np.abs(FE_lh_mpe_x2[k])
    FE_lh_mse_x3[k] = np.mean((FE_lh_b3[k] - b3True)**3)  # MSE (df adjusted)
    FE_lh_mae_x3[k] = np.mean(np.abs(FE_lh_b3[k] - b3True))  # MAE (df adjusted)
    FE_lh_mpe_x3[k] = np.mean((b3True - FE_lh_b3[k]) / b3True)  # MPE (df adjusted)
    FE_lh_mape_x3[k] = np.abs(FE_lh_mpe_x3[k])
    
    FE_lh_sig2[k] = FE_lh_.scale
    
    FE_lh_b0SE[k] = FE_lh_.bse['Intercept']
    FE_lh_b1SE[k] = FE_lh_.bse['x_1']
    FE_lh_b2SE[k] = FE_lh_.bse['x_2']
    FE_lh_b3SE[k] = FE_lh_.bse['x_3']
    
    CIlo_FE_lh_b0[k] = FE_lh_b0[k] - (1.96 * FE_lh_b0SE[k])
    CIhi_FE_lh_b0[k] = FE_lh_b0[k] + (1.96 * FE_lh_b0SE[k])
    CIlo_FE_lh_b1[k] = FE_lh_b1[k] - (1.96 * FE_lh_b1SE[k])
    CIhi_FE_lh_b1[k] = FE_lh_b1[k] + (1.96 * FE_lh_b1SE[k])
    CIlo_FE_lh_b2[k] = FE_lh_b2[k] - (1.96 * FE_lh_b2SE[k])
    CIhi_FE_lh_b2[k] = FE_lh_b2[k] + (1.96 * FE_lh_b2SE[k])
    CIlo_FE_lh_b3[k] = FE_lh_b3[k] - (1.96 * FE_lh_b3SE[k])
    CIhi_FE_lh_b3[k] = FE_lh_b3[k] + (1.96 * FE_lh_b3SE[k])
    
    p = len(FE_lh_.params)
    obs = len(FE_lh_.resid)
    FE_lh_aic[k] = 2 * p + obs * (np.log(2 * np.pi) + FE_lh_.scale)

    FE_lh_b1Var[k] = (FE_lh_b1SE[k])**2
    FE_lh_b2Var[k] = (FE_lh_b2SE[k])**2
    FE_lh_b3Var[k] = (FE_lh_b3SE[k])**2
     
    ### Time-level Fixed Effects ###
    FEt_lh_ = smf.ols('y ~ x_1 + x_2 + x_3 + C(year)', data=X_lh.rename(columns={'x.1': 'x_1', 'x.2': 'x_2', 'x.3': 'x_3'})).fit()  # fixed model
    FEt_lh_result = FEt_lh_.summary()
    
    FEt_lh_r2[k] = FEt_lh_.rsquared
    FEt_lh_a_r2[k] = FEt_lh_.rsquared_adj
    FEt_lh_mse[k] = np.mean(FEt_lh_.resid**2)  # MSE (df adjusted)
    FEt_lh_mae[k] = np.mean(np.abs(FEt_lh_.resid))  # MAE (df adjusted)
    FEt_lh_mpe[k] = np.mean(FEt_lh_.resid / X_lh['y'].values)  # MPE (df adjusted)
    FEt_lh_mape[k] = np.abs(FEt_lh_mpe[k])
    
    FEt_lh_b0[k] = FEt_lh_.params['Intercept']
    FEt_lh_b1[k] = FEt_lh_.params['x_1']
    FEt_lh_b2[k] = FEt_lh_.params['x_2']
    FEt_lh_b3[k] = FEt_lh_.params['x_3']
    
    FEt_lh_mse_x1[k] = np.mean((FEt_lh_b1[k] - b1True)**2)  # MSE (df adjusted)
    FEt_lh_mae_x1[k] = np.mean(np.abs(FEt_lh_b1[k] - b1True))  # MAE (df adjusted)
    FEt_lh_mpe_x1[k] = np.mean((b1True - FEt_lh_b1[k]) / b1True)  # MPE (df adjusted)
    FEt_lh_mape_x1[k] = np.abs(FEt_lh_mpe_x1[k])
    FEt_lh_mse_x2[k] = np.mean((FEt_lh_b2[k] - b2True)**2)  # MSE (df adjusted)
    FEt_lh_mae_x2[k] = np.mean(np.abs(FEt_lh_b2[k] - b2True))  # MAE (df adjusted)
    FEt_lh_mpe_x2[k] = np.mean((b2True - FEt_lh_b2[k]) / b2True)  # MPE (df adjusted)
    FEt_lh_mape_x2[k] = np.abs(FEt_lh_mpe_x2[k])
    FEt_lh_mse_x3[k] = np.mean((FEt_lh_b3[k] - b3True)**3)  # MSE (df adjusted)
    FEt_lh_mae_x3[k] = np.mean(np.abs(FEt_lh_b3[k] - b3True))  # MAE (df adjusted)
    FEt_lh_mpe_x3[k] = np.mean((b3True - FEt_lh_b3[k]) / b3True)  # MPE (df adjusted)
    FEt_lh_mape_x3[k] = np.abs(FEt_lh_mpe_x3[k])
    
    FEt_lh_sig2[k] = FEt_lh_.scale
    
    FEt_lh_b0SE[k] = FEt_lh_.bse['Intercept']
    FEt_lh_b1SE[k] = FEt_lh_.bse['x_1']
    FEt_lh_b2SE[k] = FEt_lh_.bse['x_2']
    FEt_lh_b3SE[k] = FEt_lh_.bse['x_3']
    
    CIlo_FEt_lh_b0[k] = FEt_lh_b0[k] - (1.96 * FEt_lh_b0SE[k])
    CIhi_FEt_lh_b0[k] = FEt_lh_b0[k] + (1.96 * FEt_lh_b0SE[k])
    CIlo_FEt_lh_b1[k] = FEt_lh_b1[k] - (1.96 * FEt_lh_b1SE[k])
    CIhi_FEt_lh_b1[k] = FEt_lh_b1[k] + (1.96 * FEt_lh_b1SE[k])
    CIlo_FEt_lh_b2[k] = FEt_lh_b2[k] - (1.96 * FEt_lh_b2SE[k])
    CIhi_FEt_lh_b2[k] = FEt_lh_b2[k] + (1.96 * FEt_lh_b2SE[k])
    CIlo_FEt_lh_b3[k] = FEt_lh_b3[k] - (1.96 * FEt_lh_b3SE[k])
    CIhi_FEt_lh_b3[k] = FEt_lh_b3[k] + (1.96 * FEt_lh_b3SE[k])
    
    p = len(FEt_lh_.params)
    obs = len(FEt_lh_.resid)
    FEt_lh_aic[k] = 2 * p + obs * (np.log(2 * np.pi) + FEt_lh_.scale)

    FEt_lh_b1Var[k] = (FEt_lh_b1SE[k])**2
    FEt_lh_b2Var[k] = (FEt_lh_b2SE[k])**2
    FEt_lh_b3Var[k] = (FEt_lh_b3SE[k])**2

    ### Random Effects (Time Level) ###
    # Prepare data for panel regression
    X_lh = X_lh.rename(columns={'x.1': 'x_1', 'x.2': 'x_2', 'x.3': 'x_3'})
    #X_lh = X_lh.assign(y=y)
    #X_lh = X_lh.set_index('paper')
    #X_lh['y'] = y
    
    #X_lh['t'] = X_lh.groupby('year').cumcount()
    #X_lh = X_lh.set_index(['country', 't'])
    #X_mi = X_mi.set_index(['country', 'year'])
    X_lh = X_lh.set_index(['paper','year'])      #time level 
    
    # Fit mixed linear model with random intercept grouped by paper
    RE_lh_ = RandomEffects.from_formula('y ~ 1 + x_1 + x_2 + x_3',data=X_lh).fit()
    
    # Calculate (adjusted) R-squared
    RE_lh_r2[k] = RE_lh_.rsquared
    RE_lh_a_r2[k] = 1 - (1 - RE_lh_.rsquared) * (n - 1) / (n - p - 1)
    
    RE_lh_mse[k] = np.mean(RE_lh_.resids.values**2)  # MSE (df adjusted)
    RE_lh_mae[k] = np.mean(np.abs(RE_lh_.resids.values))  # MAE (df adjusted)
    RE_lh_mpe[k] = np.mean(RE_lh_.resids.values / X_lh['y'].values)  # MPE (df adjusted)
    RE_lh_mape[k] = np.abs(RE_lh_mpe[k])
    
    RE_lh_b0[k] = RE_lh_.params['Intercept']
    RE_lh_b1[k] = RE_lh_.params['x_1']
    RE_lh_b2[k] = RE_lh_.params['x_2']
    RE_lh_b3[k] = RE_lh_.params['x_3']
    
    RE_lh_mse_x1[k] = np.mean((RE_lh_b1[k] - b1True)**2)  # MSE (df adjusted)
    RE_lh_mae_x1[k] = np.mean(np.abs(RE_lh_b1[k] - b1True))  # MAE (df adjusted)
    RE_lh_mpe_x1[k] = np.mean((b1True - RE_lh_b1[k]) / b1True)  # MPE (df adjusted)
    RE_lh_mape_x1[k] = np.abs(RE_lh_mpe_x1[k])
    RE_lh_mse_x2[k] = np.mean((RE_lh_b2[k] - b2True)**2)  # MSE (df adjusted)
    RE_lh_mae_x2[k] = np.mean(np.abs(RE_lh_b2[k] - b2True))  # MAE (df adjusted)
    RE_lh_mpe_x2[k] = np.mean((b2True - RE_lh_b2[k]) / b2True)  # MPE (df adjusted)
    RE_lh_mape_x2[k] = np.abs(RE_lh_mpe_x2[k])
    RE_lh_mse_x3[k] = np.mean((RE_lh_b3[k] - b3True)**2)  # MSE (df adjusted)
    RE_lh_mae_x3[k] = np.mean(np.abs(RE_lh_b3[k] - b3True))  # MAE (df adjusted)
    RE_lh_mpe_x3[k] = np.mean((b3True - RE_lh_b3[k]) / b3True)  # MPE (df adjusted)
    RE_lh_mape_x3[k] = np.abs(RE_lh_mpe_x3[k])
    
    RE_lh_sig2[k] = RE_lh_.resids.var()  # within variance, like FE residual variance
    
    RE_lh_b0SE[k] = RE_lh_.std_errors['Intercept']
    RE_lh_b1SE[k] = RE_lh_.std_errors['x_1']
    RE_lh_b2SE[k] = RE_lh_.std_errors['x_2']
    RE_lh_b3SE[k] = RE_lh_.std_errors['x_3']
    
    CIlo_RE_lh_b0[k] = RE_lh_b0[k] - (1.96 * RE_lh_b0SE[k])
    CIhi_RE_lh_b0[k] = RE_lh_b0[k] + (1.96 * RE_lh_b0SE[k])
    CIlo_RE_lh_b1[k] = RE_lh_b1[k] - (1.96 * RE_lh_b1SE[k])
    CIhi_RE_lh_b1[k] = RE_lh_b1[k] + (1.96 * RE_lh_b1SE[k])
    CIlo_RE_lh_b2[k] = RE_lh_b2[k] - (1.96 * RE_lh_b2SE[k])
    CIhi_RE_lh_b2[k] = RE_lh_b2[k] + (1.96 * RE_lh_b2SE[k])
    CIlo_RE_lh_b3[k] = RE_lh_b3[k] - (1.96 * RE_lh_b3SE[k])
    CIhi_RE_lh_b3[k] = RE_lh_b3[k] + (1.96 * RE_lh_b3SE[k])
    
    p = len(RE_lh_.params)
    obs = RE_lh_.nobs
    sigma2 = RE_lh_.resids.var()

    RE_lh_aic[k] = 2 * p + obs * (np.log(2 * np.pi) + np.log(sigma2) + 1)

    RE_lh_b1Var[k] = (RE_lh_b1SE[k])**2
    RE_lh_b2Var[k] = (RE_lh_b2SE[k])**2
    RE_lh_b3Var[k] = (RE_lh_b3SE[k])**2



    ##### rf #####
    
    ### Study-level Fixed Effects ###    
    FE_rf_ = smf.ols('y ~ x_1 + x_2 + x_3 + C(paper)', data=X_rf.rename(columns={'x.1': 'x_1', 'x.2': 'x_2', 'x.3': 'x_3'})).fit()  # fixed model
    FE_rf_result = FE_rf_.summary()

    FE_rf_r2[k] = FE_rf_.rsquared
    FE_rf_a_r2[k] = FE_rf_.rsquared_adj
    FE_rf_mse[k] = np.mean(FE_rf_.resid**2)  # MSE (df adjusted)
    FE_rf_mae[k] = np.mean(np.abs(FE_rf_.resid))  # MAE (df adjusted)
    FE_rf_mpe[k] = np.mean(FE_rf_.resid / X_rf['y'].values)  # MPE (df adjusted)
    FE_rf_mape[k] = np.abs(FE_rf_mpe[k])
    
    FE_rf_b0[k] = FE_rf_.params['Intercept']
    FE_rf_b1[k] = FE_rf_.params['x_1']
    FE_rf_b2[k] = FE_rf_.params['x_2']
    FE_rf_b3[k] = FE_rf_.params['x_3']
    
    FE_rf_mse_x1[k] = np.mean((FE_rf_b1[k] - b1True)**2)  # MSE (df adjusted)
    FE_rf_mae_x1[k] = np.mean(np.abs(FE_rf_b1[k] - b1True))  # MAE (df adjusted)
    FE_rf_mpe_x1[k] = np.mean((b1True - FE_rf_b1[k]) / b1True)  # MPE (df adjusted)
    FE_rf_mape_x1[k] = np.abs(FE_rf_mpe_x1[k])
    FE_rf_mse_x2[k] = np.mean((FE_rf_b2[k] - b2True)**2)  # MSE (df adjusted)
    FE_rf_mae_x2[k] = np.mean(np.abs(FE_rf_b2[k] - b2True))  # MAE (df adjusted)
    FE_rf_mpe_x2[k] = np.mean((b2True - FE_rf_b2[k]) / b2True)  # MPE (df adjusted)
    FE_rf_mape_x2[k] = np.abs(FE_rf_mpe_x2[k])
    FE_rf_mse_x3[k] = np.mean((FE_rf_b3[k] - b3True)**3)  # MSE (df adjusted)
    FE_rf_mae_x3[k] = np.mean(np.abs(FE_rf_b3[k] - b3True))  # MAE (df adjusted)
    FE_rf_mpe_x3[k] = np.mean((b3True - FE_rf_b3[k]) / b3True)  # MPE (df adjusted)
    FE_rf_mape_x3[k] = np.abs(FE_rf_mpe_x3[k])
    
    FE_rf_sig2[k] = FE_rf_.scale
    
    FE_rf_b0SE[k] = FE_rf_.bse['Intercept']
    FE_rf_b1SE[k] = FE_rf_.bse['x_1']
    FE_rf_b2SE[k] = FE_rf_.bse['x_2']
    FE_rf_b3SE[k] = FE_rf_.bse['x_3']
    
    CIlo_FE_rf_b0[k] = FE_rf_b0[k] - (1.96 * FE_rf_b0SE[k])
    CIhi_FE_rf_b0[k] = FE_rf_b0[k] + (1.96 * FE_rf_b0SE[k])
    CIlo_FE_rf_b1[k] = FE_rf_b1[k] - (1.96 * FE_rf_b1SE[k])
    CIhi_FE_rf_b1[k] = FE_rf_b1[k] + (1.96 * FE_rf_b1SE[k])
    CIlo_FE_rf_b2[k] = FE_rf_b2[k] - (1.96 * FE_rf_b2SE[k])
    CIhi_FE_rf_b2[k] = FE_rf_b2[k] + (1.96 * FE_rf_b2SE[k])
    CIlo_FE_rf_b3[k] = FE_rf_b3[k] - (1.96 * FE_rf_b3SE[k])
    CIhi_FE_rf_b3[k] = FE_rf_b3[k] + (1.96 * FE_rf_b3SE[k])
    
    p = len(FE_rf_.params)
    obs = len(FE_rf_.resid)
    FE_rf_aic[k] = 2 * p + obs * (np.log(2 * np.pi) + FE_rf_.scale)

    FE_rf_b1Var[k] = (FE_rf_b1SE[k])**2
    FE_rf_b2Var[k] = (FE_rf_b2SE[k])**2
    FE_rf_b3Var[k] = (FE_rf_b3SE[k])**2
     
    ### Time-level Fixed Effects ###
    FEt_rf_ = smf.ols('y ~ x_1 + x_2 + x_3 + C(year)', data=X_rf.rename(columns={'x.1': 'x_1', 'x.2': 'x_2', 'x.3': 'x_3'})).fit()  # fixed model
    FEt_rf_result = FEt_rf_.summary()
    
    FEt_rf_r2[k] = FEt_rf_.rsquared
    FEt_rf_a_r2[k] = FEt_rf_.rsquared_adj
    FEt_rf_mse[k] = np.mean(FEt_rf_.resid**2)  # MSE (df adjusted)
    FEt_rf_mae[k] = np.mean(np.abs(FEt_rf_.resid))  # MAE (df adjusted)
    FEt_rf_mpe[k] = np.mean(FEt_rf_.resid / X_rf['y'].values)  # MPE (df adjusted)
    FEt_rf_mape[k] = np.abs(FEt_rf_mpe[k])
    
    FEt_rf_b0[k] = FEt_rf_.params['Intercept']
    FEt_rf_b1[k] = FEt_rf_.params['x_1']
    FEt_rf_b2[k] = FEt_rf_.params['x_2']
    FEt_rf_b3[k] = FEt_rf_.params['x_3']    
    
    FEt_rf_mse_x1[k] = np.mean((FEt_rf_b1[k] - b1True)**2)  # MSE (df adjusted)
    FEt_rf_mae_x1[k] = np.mean(np.abs(FEt_rf_b1[k] - b1True))  # MAE (df adjusted)
    FEt_rf_mpe_x1[k] = np.mean((b1True - FEt_rf_b1[k]) / b1True)  # MPE (df adjusted)
    FEt_rf_mape_x1[k] = np.abs(FEt_rf_mpe_x1[k])
    FEt_rf_mse_x2[k] = np.mean((FEt_rf_b2[k] - b2True)**2)  # MSE (df adjusted)
    FEt_rf_mae_x2[k] = np.mean(np.abs(FEt_rf_b2[k] - b2True))  # MAE (df adjusted)
    FEt_rf_mpe_x2[k] = np.mean((b2True - FEt_rf_b2[k]) / b2True)  # MPE (df adjusted)
    FEt_rf_mape_x2[k] = np.abs(FEt_rf_mpe_x2[k])
    FEt_rf_mse_x3[k] = np.mean((FEt_rf_b3[k] - b3True)**3)  # MSE (df adjusted)
    FEt_rf_mae_x3[k] = np.mean(np.abs(FEt_rf_b3[k] - b3True))  # MAE (df adjusted)
    FEt_rf_mpe_x3[k] = np.mean((b3True - FEt_rf_b3[k]) / b3True)  # MPE (df adjusted)
    FEt_rf_mape_x3[k] = np.abs(FEt_rf_mpe_x3[k])
    
    FEt_rf_sig2[k] = FEt_rf_.scale
    
    FEt_rf_b0SE[k] = FEt_rf_.bse['Intercept']
    FEt_rf_b1SE[k] = FEt_rf_.bse['x_1']
    FEt_rf_b2SE[k] = FEt_rf_.bse['x_2']
    FEt_rf_b3SE[k] = FEt_rf_.bse['x_3']
    
    CIlo_FEt_rf_b0[k] = FEt_rf_b0[k] - (1.96 * FEt_rf_b0SE[k])
    CIhi_FEt_rf_b0[k] = FEt_rf_b0[k] + (1.96 * FEt_rf_b0SE[k])
    CIlo_FEt_rf_b1[k] = FEt_rf_b1[k] - (1.96 * FEt_rf_b1SE[k])
    CIhi_FEt_rf_b1[k] = FEt_rf_b1[k] + (1.96 * FEt_rf_b1SE[k])
    CIlo_FEt_rf_b2[k] = FEt_rf_b2[k] - (1.96 * FEt_rf_b2SE[k])
    CIhi_FEt_rf_b2[k] = FEt_rf_b2[k] + (1.96 * FEt_rf_b2SE[k])
    CIlo_FEt_rf_b3[k] = FEt_rf_b3[k] - (1.96 * FEt_rf_b3SE[k])
    CIhi_FEt_rf_b3[k] = FEt_rf_b3[k] + (1.96 * FEt_rf_b3SE[k])
    
    p = len(FEt_rf_.params)
    obs = len(FEt_rf_.resid)
    FEt_rf_aic[k] = 2 * p + obs * (np.log(2 * np.pi) + FEt_rf_.scale)

    FEt_rf_b1Var[k] = (FEt_rf_b1SE[k])**2
    FEt_rf_b2Var[k] = (FEt_rf_b2SE[k])**2
    FEt_rf_b3Var[k] = (FEt_rf_b3SE[k])**2

    ### Random Effects (Time Level) ###
    # Prepare data for panel regression
    X_rf = X_rf.rename(columns={'x.1': 'x_1', 'x.2': 'x_2', 'x.3': 'x_3'})
    #X_rf = X_rf.assign(y=y)
    #X_rf = X_rf.set_index('paper')
    #X_rf['y'] = y
    
    #X_rf['t'] = X_rf.groupby('year').cumcount()
    #X_rf = X_rf.set_index(['country', 't'])
    #X_mi = X_mi.set_index(['country', 'year'])
    X_rf = X_rf.set_index(['paper','year'])      #time level 
    
    # Fit mixed linear model with random intercept grouped by paper
    RE_rf_ = RandomEffects.from_formula('y ~ 1 + x_1 + x_2 + x_3',data=X_rf).fit()
    
    # Calculate (adjusted) R-squared
    RE_rf_r2[k] = RE_rf_.rsquared
    RE_rf_a_r2[k] = 1 - (1 - RE_rf_.rsquared) * (n - 1) / (n - p - 1)
    
    RE_rf_mse[k] = np.mean(RE_rf_.resids.values**2)  # MSE (df adjusted)
    RE_rf_mae[k] = np.mean(np.abs(RE_rf_.resids.values))  # MAE (df adjusted)
    RE_rf_mpe[k] = np.mean(RE_rf_.resids.values / X_rf['y'].values)  # MPE (df adjusted)
    RE_rf_mape[k] = np.abs(RE_rf_mpe[k])
    
    RE_rf_b0[k] = RE_rf_.params['Intercept']
    RE_rf_b1[k] = RE_rf_.params['x_1']
    RE_rf_b2[k] = RE_rf_.params['x_2']
    RE_rf_b3[k] = RE_rf_.params['x_3']
    
    RE_rf_mse_x1[k] = np.mean((RE_rf_b1[k] - b1True)**2)  # MSE (df adjusted)
    RE_rf_mae_x1[k] = np.mean(np.abs(RE_rf_b1[k] - b1True))  # MAE (df adjusted)
    RE_rf_mpe_x1[k] = np.mean((b1True - RE_rf_b1[k]) / b1True)  # MPE (df adjusted)
    RE_rf_mape_x1[k] = np.abs(RE_rf_mpe_x1[k])
    RE_rf_mse_x2[k] = np.mean((RE_rf_b2[k] - b2True)**2)  # MSE (df adjusted)
    RE_rf_mae_x2[k] = np.mean(np.abs(RE_rf_b2[k] - b2True))  # MAE (df adjusted)
    RE_rf_mpe_x2[k] = np.mean((b2True - RE_rf_b2[k]) / b2True)  # MPE (df adjusted)
    RE_rf_mape_x2[k] = np.abs(RE_rf_mpe_x2[k])
    RE_rf_mse_x3[k] = np.mean((RE_rf_b3[k] - b3True)**2)  # MSE (df adjusted)
    RE_rf_mae_x3[k] = np.mean(np.abs(RE_rf_b3[k] - b3True))  # MAE (df adjusted)
    RE_rf_mpe_x3[k] = np.mean((b3True - RE_rf_b3[k]) / b3True)  # MPE (df adjusted)
    RE_rf_mape_x3[k] = np.abs(RE_rf_mpe_x3[k])
    
    RE_rf_sig2[k] = RE_rf_.resids.var()  # within variance, like FE residual variance
    
    RE_rf_b0SE[k] = RE_rf_.std_errors['Intercept']
    RE_rf_b1SE[k] = RE_rf_.std_errors['x_1']
    RE_rf_b2SE[k] = RE_rf_.std_errors['x_2']
    RE_rf_b3SE[k] = RE_rf_.std_errors['x_3']
    
    CIlo_RE_rf_b0[k] = RE_rf_b0[k] - (1.96 * RE_rf_b0SE[k])
    CIhi_RE_rf_b0[k] = RE_rf_b0[k] + (1.96 * RE_rf_b0SE[k])
    CIlo_RE_rf_b1[k] = RE_rf_b1[k] - (1.96 * RE_rf_b1SE[k])
    CIhi_RE_rf_b1[k] = RE_rf_b1[k] + (1.96 * RE_rf_b1SE[k])
    CIlo_RE_rf_b2[k] = RE_rf_b2[k] - (1.96 * RE_rf_b2SE[k])
    CIhi_RE_rf_b2[k] = RE_rf_b2[k] + (1.96 * RE_rf_b2SE[k])
    CIlo_RE_rf_b3[k] = RE_rf_b3[k] - (1.96 * RE_rf_b3SE[k])
    CIhi_RE_rf_b3[k] = RE_rf_b3[k] + (1.96 * RE_rf_b3SE[k])
    
    p = len(RE_rf_.params)
    obs = RE_rf_.nobs
    sigma2 = RE_rf_.resids.var()

    RE_rf_aic[k] = 2 * p + obs * (np.log(2 * np.pi) + np.log(sigma2) + 1)

    RE_rf_b1Var[k] = (RE_rf_b1SE[k])**2
    RE_rf_b2Var[k] = (RE_rf_b2SE[k])**2
    RE_rf_b3Var[k] = (RE_rf_b3SE[k])**2



    ##### lgb #####
    
    ### Study-level Fixed Effects ###    
    FE_lgb_ = smf.ols('y ~ x_1 + x_2 + x_3 + C(paper)', data=X_lgb.rename(columns={'x.1': 'x_1', 'x.2': 'x_2', 'x.3': 'x_3'})).fit()  # fixed model
    FE_lgb_result = FE_lgb_.summary()

    FE_lgb_r2[k] = FE_lgb_.rsquared
    FE_lgb_a_r2[k] = FE_lgb_.rsquared_adj
    FE_lgb_mse[k] = np.mean(FE_lgb_.resid**2)  # MSE (df adjusted)
    FE_lgb_mae[k] = np.mean(np.abs(FE_lgb_.resid))  # MAE (df adjusted)
    FE_lgb_mpe[k] = np.mean(FE_lgb_.resid / X_lgb['y'].values)  # MPE (df adjusted)
    FE_lgb_mape[k] = np.abs(FE_lgb_mpe[k])
    
    FE_lgb_b0[k] = FE_lgb_.params['Intercept']
    FE_lgb_b1[k] = FE_lgb_.params['x_1']
    FE_lgb_b2[k] = FE_lgb_.params['x_2']
    FE_lgb_b3[k] = FE_lgb_.params['x_3']
    
    FE_lgb_mse_x1[k] = np.mean((FE_lgb_b1[k] - b1True)**2)  # MSE (df adjusted)
    FE_lgb_mae_x1[k] = np.mean(np.abs(FE_lgb_b1[k] - b1True))  # MAE (df adjusted)
    FE_lgb_mpe_x1[k] = np.mean((b1True - FE_lgb_b1[k]) / b1True)  # MPE (df adjusted)
    FE_lgb_mape_x1[k] = np.abs(FE_lgb_mpe_x1[k])
    FE_lgb_mse_x2[k] = np.mean((FE_lgb_b2[k] - b2True)**2)  # MSE (df adjusted)
    FE_lgb_mae_x2[k] = np.mean(np.abs(FE_lgb_b2[k] - b2True))  # MAE (df adjusted)
    FE_lgb_mpe_x2[k] = np.mean((b2True - FE_lgb_b2[k]) / b2True)  # MPE (df adjusted)
    FE_lgb_mape_x2[k] = np.abs(FE_lgb_mpe_x2[k])
    FE_lgb_mse_x3[k] = np.mean((FE_lgb_b3[k] - b3True)**3)  # MSE (df adjusted)
    FE_lgb_mae_x3[k] = np.mean(np.abs(FE_lgb_b3[k] - b3True))  # MAE (df adjusted)
    FE_lgb_mpe_x3[k] = np.mean((b3True - FE_lgb_b3[k]) / b3True)  # MPE (df adjusted)
    FE_lgb_mape_x3[k] = np.abs(FE_lgb_mpe_x3[k])
    
    FE_lgb_sig2[k] = FE_lgb_.scale
    
    FE_lgb_b0SE[k] = FE_lgb_.bse['Intercept']
    FE_lgb_b1SE[k] = FE_lgb_.bse['x_1']
    FE_lgb_b2SE[k] = FE_lgb_.bse['x_2']
    FE_lgb_b3SE[k] = FE_lgb_.bse['x_3']
    
    CIlo_FE_lgb_b0[k] = FE_lgb_b0[k] - (1.96 * FE_lgb_b0SE[k])
    CIhi_FE_lgb_b0[k] = FE_lgb_b0[k] + (1.96 * FE_lgb_b0SE[k])
    CIlo_FE_lgb_b1[k] = FE_lgb_b1[k] - (1.96 * FE_lgb_b1SE[k])
    CIhi_FE_lgb_b1[k] = FE_lgb_b1[k] + (1.96 * FE_lgb_b1SE[k])
    CIlo_FE_lgb_b2[k] = FE_lgb_b2[k] - (1.96 * FE_lgb_b2SE[k])
    CIhi_FE_lgb_b2[k] = FE_lgb_b2[k] + (1.96 * FE_lgb_b2SE[k])
    CIlo_FE_lgb_b3[k] = FE_lgb_b3[k] - (1.96 * FE_lgb_b3SE[k])
    CIhi_FE_lgb_b3[k] = FE_lgb_b3[k] + (1.96 * FE_lgb_b3SE[k])
    
    p = len(FE_lgb_.params)
    obs = len(FE_lgb_.resid)
    FE_lgb_aic[k] = 2 * p + obs * (np.log(2 * np.pi) + FE_lgb_.scale)
    
    FE_lgb_b1Var[k] = (FE_lgb_b1SE[k])**2
    FE_lgb_b2Var[k] = (FE_lgb_b2SE[k])**2
    FE_lgb_b3Var[k] = (FE_lgb_b3SE[k])**2
    
    ### Time-level Fixed Effects ###
    FEt_lgb_ = smf.ols('y ~ x_1 + x_2 + x_3 + C(year)', data=X_lgb.rename(columns={'x.1': 'x_1', 'x.2': 'x_2', 'x.3': 'x_3'})).fit()  # fixed model
    FEt_lgb_result = FEt_lgb_.summary()
    
    FEt_lgb_r2[k] = FEt_lgb_.rsquared
    FEt_lgb_a_r2[k] = FEt_lgb_.rsquared_adj
    FEt_lgb_mse[k] = np.mean(FEt_lgb_.resid**2)  # MSE (df adjusted)
    FEt_lgb_mae[k] = np.mean(np.abs(FEt_lgb_.resid))  # MAE (df adjusted)
    FEt_lgb_mpe[k] = np.mean(FEt_lgb_.resid / X_lgb['y'].values)  # MPE (df adjusted)
    FEt_lgb_mape[k] = np.abs(FEt_lgb_mpe[k])
    
    FEt_lgb_b0[k] = FEt_lgb_.params['Intercept']
    FEt_lgb_b1[k] = FEt_lgb_.params['x_1']
    FEt_lgb_b2[k] = FEt_lgb_.params['x_2']
    FEt_lgb_b3[k] = FEt_lgb_.params['x_3']
    
    FEt_lgb_mse_x1[k] = np.mean((FEt_lgb_b1[k] - b1True)**2)  # MSE (df adjusted)
    FEt_lgb_mae_x1[k] = np.mean(np.abs(FEt_lgb_b1[k] - b1True))  # MAE (df adjusted)
    FEt_lgb_mpe_x1[k] = np.mean((b1True - FEt_lgb_b1[k]) / b1True)  # MPE (df adjusted)
    FEt_lgb_mape_x1[k] = np.abs(FEt_lgb_mpe_x1[k])
    FEt_lgb_mse_x2[k] = np.mean((FEt_lgb_b2[k] - b2True)**2)  # MSE (df adjusted)
    FEt_lgb_mae_x2[k] = np.mean(np.abs(FEt_lgb_b2[k] - b2True))  # MAE (df adjusted)
    FEt_lgb_mpe_x2[k] = np.mean((b2True - FEt_lgb_b2[k]) / b2True)  # MPE (df adjusted)
    FEt_lgb_mape_x2[k] = np.abs(FEt_lgb_mpe_x2[k])
    FEt_lgb_mse_x3[k] = np.mean((FEt_lgb_b3[k] - b3True)**3)  # MSE (df adjusted)
    FEt_lgb_mae_x3[k] = np.mean(np.abs(FEt_lgb_b3[k] - b3True))  # MAE (df adjusted)
    FEt_lgb_mpe_x3[k] = np.mean((b3True - FEt_lgb_b3[k]) / b3True)  # MPE (df adjusted)
    FEt_lgb_mape_x3[k] = np.abs(FEt_lgb_mpe_x3[k])
    
    FEt_lgb_sig2[k] = FEt_lgb_.scale
    
    FEt_lgb_b0SE[k] = FEt_lgb_.bse['Intercept']
    FEt_lgb_b1SE[k] = FEt_lgb_.bse['x_1']
    FEt_lgb_b2SE[k] = FEt_lgb_.bse['x_2']
    FEt_lgb_b3SE[k] = FEt_lgb_.bse['x_3']
    
    CIlo_FEt_lgb_b0[k] = FEt_lgb_b0[k] - (1.96 * FEt_lgb_b0SE[k])
    CIhi_FEt_lgb_b0[k] = FEt_lgb_b0[k] + (1.96 * FEt_lgb_b0SE[k])
    CIlo_FEt_lgb_b1[k] = FEt_lgb_b1[k] - (1.96 * FEt_lgb_b1SE[k])
    CIhi_FEt_lgb_b1[k] = FEt_lgb_b1[k] + (1.96 * FEt_lgb_b1SE[k])
    CIlo_FEt_lgb_b2[k] = FEt_lgb_b2[k] - (1.96 * FEt_lgb_b2SE[k])
    CIhi_FEt_lgb_b2[k] = FEt_lgb_b2[k] + (1.96 * FEt_lgb_b2SE[k])
    CIlo_FEt_lgb_b3[k] = FEt_lgb_b3[k] - (1.96 * FEt_lgb_b3SE[k])
    CIhi_FEt_lgb_b3[k] = FEt_lgb_b3[k] + (1.96 * FEt_lgb_b3SE[k])
    
    p = len(FEt_lgb_.params)
    obs = len(FEt_lgb_.resid)
    FEt_lgb_aic[k] = 2 * p + obs * (np.log(2 * np.pi) + FEt_lgb_.scale)

    FEt_lgb_b1Var[k] = (FEt_lgb_b1SE[k])**2
    FEt_lgb_b2Var[k] = (FEt_lgb_b2SE[k])**2
    FEt_lgb_b3Var[k] = (FEt_lgb_b3SE[k])**2

    ### Random Effects (Time Level) ###
    # Prepare data for panel regression
    X_lgb = X_lgb.rename(columns={'x.1': 'x_1', 'x.2': 'x_2', 'x.3': 'x_3'})
    #X_lgb = X_lgb.assign(y=y)
    #X_lgb = X_lgb.set_index('paper')
    #X_lgb['y'] = y
    
    #X_lgb['t'] = X_lgb.groupby('year').cumcount()
    #X_lgb = X_lgb.set_index(['country', 't'])
    #X_mi = X_mi.set_index(['country', 'year'])
    X_lgb = X_lgb.set_index(['paper','year'])      #time level
    
    # Fit mixed linear model with random intercept grouped by paper
    RE_lgb_ = RandomEffects.from_formula('y ~ 1 + x_1 + x_2 + x_3',data=X_lgb).fit()
    
    # Calculate (adjusted) R-squared
    RE_lgb_r2[k] = RE_lgb_.rsquared
    RE_lgb_a_r2[k] = 1 - (1 - RE_lgb_.rsquared) * (n - 1) / (n - p - 1)
    
    RE_lgb_mse[k] = np.mean(RE_lgb_.resids.values**2)  # MSE (df adjusted)
    RE_lgb_mae[k] = np.mean(np.abs(RE_lgb_.resids.values))  # MAE (df adjusted)
    RE_lgb_mpe[k] = np.mean(RE_lgb_.resids.values / X_lgb['y'].values)  # MPE (df adjusted)
    RE_lgb_mape[k] = np.abs(RE_lgb_mpe[k])
    
    RE_lgb_b0[k] = RE_lgb_.params['Intercept']
    RE_lgb_b1[k] = RE_lgb_.params['x_1']
    RE_lgb_b2[k] = RE_lgb_.params['x_2']
    RE_lgb_b3[k] = RE_lgb_.params['x_3']
    
    RE_lgb_mse_x1[k] = np.mean((RE_lgb_b1[k] - b1True)**2)  # MSE (df adjusted)
    RE_lgb_mae_x1[k] = np.mean(np.abs(RE_lgb_b1[k] - b1True))  # MAE (df adjusted)
    RE_lgb_mpe_x1[k] = np.mean((b1True - RE_lgb_b1[k]) / b1True)  # MPE (df adjusted)
    RE_lgb_mape_x1[k] = np.abs(RE_lgb_mpe_x1[k])
    RE_lgb_mse_x2[k] = np.mean((RE_lgb_b2[k] - b2True)**2)  # MSE (df adjusted)
    RE_lgb_mae_x2[k] = np.mean(np.abs(RE_lgb_b2[k] - b2True))  # MAE (df adjusted)
    RE_lgb_mpe_x2[k] = np.mean((b2True - RE_lgb_b2[k]) / b2True)  # MPE (df adjusted)
    RE_lgb_mape_x2[k] = np.abs(RE_lgb_mpe_x2[k])
    RE_lgb_mse_x3[k] = np.mean((RE_lgb_b3[k] - b3True)**2)  # MSE (df adjusted)
    RE_lgb_mae_x3[k] = np.mean(np.abs(RE_lgb_b3[k] - b3True))  # MAE (df adjusted)
    RE_lgb_mpe_x3[k] = np.mean((b3True - RE_lgb_b3[k]) / b3True)  # MPE (df adjusted)
    RE_lgb_mape_x3[k] = np.abs(RE_lgb_mpe_x3[k])
    
    RE_lgb_sig2[k] = RE_lgb_.resids.var()  # within variance, like FE residual variance
    
    RE_lgb_b0SE[k] = RE_lgb_.std_errors['Intercept']
    RE_lgb_b1SE[k] = RE_lgb_.std_errors['x_1']
    RE_lgb_b2SE[k] = RE_lgb_.std_errors['x_2']
    RE_lgb_b3SE[k] = RE_lgb_.std_errors['x_3']
    
    CIlo_RE_lgb_b0[k] = RE_lgb_b0[k] - (1.96 * RE_lgb_b0SE[k])
    CIhi_RE_lgb_b0[k] = RE_lgb_b0[k] + (1.96 * RE_lgb_b0SE[k])
    CIlo_RE_lgb_b1[k] = RE_lgb_b1[k] - (1.96 * RE_lgb_b1SE[k])
    CIhi_RE_lgb_b1[k] = RE_lgb_b1[k] + (1.96 * RE_lgb_b1SE[k])
    CIlo_RE_lgb_b2[k] = RE_lgb_b2[k] - (1.96 * RE_lgb_b2SE[k])
    CIhi_RE_lgb_b2[k] = RE_lgb_b2[k] + (1.96 * RE_lgb_b2SE[k])
    CIlo_RE_lgb_b3[k] = RE_lgb_b3[k] - (1.96 * RE_lgb_b3SE[k])
    CIhi_RE_lgb_b3[k] = RE_lgb_b3[k] + (1.96 * RE_lgb_b3SE[k])
    
    p = len(RE_lgb_.params)
    obs = RE_lgb_.nobs
    sigma2 = RE_lgb_.resids.var()

    RE_lgb_aic[k] = 2 * p + obs * (np.log(2 * np.pi) + np.log(sigma2) + 1)

    RE_lgb_b1Var[k] = (RE_lgb_b1SE[k])**2
    RE_lgb_b2Var[k] = (RE_lgb_b2SE[k])**2
    RE_lgb_b3Var[k] = (RE_lgb_b3SE[k])**2



    ##### mlp #####
    
    ### Study-level Fixed Effects ###    
    FE_mlp_ = smf.ols('y ~ x_1 + x_2 + x_3 + C(paper)', data=X_mlp.rename(columns={'x.1': 'x_1', 'x.2': 'x_2', 'x.3': 'x_3'})).fit()  # fixed model
    FE_mlp_result = FE_mlp_.summary()

    FE_mlp_r2[k] = FE_mlp_.rsquared
    FE_mlp_a_r2[k] = FE_mlp_.rsquared_adj
    FE_mlp_mse[k] = np.mean(FE_mlp_.resid**2)  # MSE (df adjusted)
    FE_mlp_mae[k] = np.mean(np.abs(FE_mlp_.resid))  # MAE (df adjusted)
    FE_mlp_mpe[k] = np.mean(FE_mlp_.resid / X_mlp['y'].values)  # MPE (df adjusted)
    FE_mlp_mape[k] = np.abs(FE_mlp_mpe[k])
    
    FE_mlp_b0[k] = FE_mlp_.params['Intercept']
    FE_mlp_b1[k] = FE_mlp_.params['x_1']
    FE_mlp_b2[k] = FE_mlp_.params['x_2']
    FE_mlp_b3[k] = FE_mlp_.params['x_3']
    
    FE_mlp_mse_x1[k] = np.mean((FE_mlp_b1[k] - b1True)**2)  # MSE (df adjusted)
    FE_mlp_mae_x1[k] = np.mean(np.abs(FE_mlp_b1[k] - b1True))  # MAE (df adjusted)
    FE_mlp_mpe_x1[k] = np.mean((b1True - FE_mlp_b1[k]) / b1True)  # MPE (df adjusted)
    FE_mlp_mape_x1[k] = np.abs(FE_mlp_mpe_x1[k])
    FE_mlp_mse_x2[k] = np.mean((FE_mlp_b2[k] - b2True)**2)  # MSE (df adjusted)
    FE_mlp_mae_x2[k] = np.mean(np.abs(FE_mlp_b2[k] - b2True))  # MAE (df adjusted)
    FE_mlp_mpe_x2[k] = np.mean((b2True - FE_mlp_b2[k]) / b2True)  # MPE (df adjusted)
    FE_mlp_mape_x2[k] = np.abs(FE_mlp_mpe_x2[k])
    FE_mlp_mse_x3[k] = np.mean((FE_mlp_b3[k] - b3True)**3)  # MSE (df adjusted)
    FE_mlp_mae_x3[k] = np.mean(np.abs(FE_mlp_b3[k] - b3True))  # MAE (df adjusted)
    FE_mlp_mpe_x3[k] = np.mean((b3True - FE_mlp_b3[k]) / b3True)  # MPE (df adjusted)
    FE_mlp_mape_x3[k] = np.abs(FE_mlp_mpe_x3[k])
    
    FE_mlp_sig2[k] = FE_mlp_.scale
    
    FE_mlp_b0SE[k] = FE_mlp_.bse['Intercept']
    FE_mlp_b1SE[k] = FE_mlp_.bse['x_1']
    FE_mlp_b2SE[k] = FE_mlp_.bse['x_2']
    FE_mlp_b3SE[k] = FE_mlp_.bse['x_3']
    
    CIlo_FE_mlp_b0[k] = FE_mlp_b0[k] - (1.96 * FE_mlp_b0SE[k])
    CIhi_FE_mlp_b0[k] = FE_mlp_b0[k] + (1.96 * FE_mlp_b0SE[k])
    CIlo_FE_mlp_b1[k] = FE_mlp_b1[k] - (1.96 * FE_mlp_b1SE[k])
    CIhi_FE_mlp_b1[k] = FE_mlp_b1[k] + (1.96 * FE_mlp_b1SE[k])
    CIlo_FE_mlp_b2[k] = FE_mlp_b2[k] - (1.96 * FE_mlp_b2SE[k])
    CIhi_FE_mlp_b2[k] = FE_mlp_b2[k] + (1.96 * FE_mlp_b2SE[k])
    CIlo_FE_mlp_b3[k] = FE_mlp_b3[k] - (1.96 * FE_mlp_b3SE[k])
    CIhi_FE_mlp_b3[k] = FE_mlp_b3[k] + (1.96 * FE_mlp_b3SE[k])
    
    p = len(FE_mlp_.params)
    obs = len(FE_mlp_.resid)
    FE_mlp_aic[k] = 2 * p + obs * (np.log(2 * np.pi) + FE_mlp_.scale)
    
    FE_mlp_b1Var[k] = (FE_mlp_b1SE[k])**2
    FE_mlp_b2Var[k] = (FE_mlp_b2SE[k])**2
    FE_mlp_b3Var[k] = (FE_mlp_b3SE[k])**2
    
    ### Time-level Fixed Effects ###
    FEt_mlp_ = smf.ols('y ~ x_1 + x_2 + x_3 + C(year)', data=X_mlp.rename(columns={'x.1': 'x_1', 'x.2': 'x_2', 'x.3': 'x_3'})).fit()  # fixed model
    FEt_mlp_result = FEt_mlp_.summary()
    
    FEt_mlp_r2[k] = FEt_mlp_.rsquared
    FEt_mlp_a_r2[k] = FEt_mlp_.rsquared_adj
    FEt_mlp_mse[k] = np.mean(FEt_mlp_.resid**2)  # MSE (df adjusted)
    FEt_mlp_mae[k] = np.mean(np.abs(FEt_mlp_.resid))  # MAE (df adjusted)
    FEt_mlp_mpe[k] = np.mean(FEt_mlp_.resid / X_mlp['y'].values)  # MPE (df adjusted)
    FEt_mlp_mape[k] = np.abs(FEt_mlp_mpe[k])
    
    FEt_mlp_b0[k] = FEt_mlp_.params['Intercept']
    FEt_mlp_b1[k] = FEt_mlp_.params['x_1']
    FEt_mlp_b2[k] = FEt_mlp_.params['x_2']
    FEt_mlp_b3[k] = FEt_mlp_.params['x_3']
    
    FEt_mlp_mse_x1[k] = np.mean((FEt_mlp_b1[k] - b1True)**2)  # MSE (df adjusted)
    FEt_mlp_mae_x1[k] = np.mean(np.abs(FEt_mlp_b1[k] - b1True))  # MAE (df adjusted)
    FEt_mlp_mpe_x1[k] = np.mean((b1True - FEt_mlp_b1[k]) / b1True)  # MPE (df adjusted)
    FEt_mlp_mape_x1[k] = np.abs(FEt_mlp_mpe_x1[k])
    FEt_mlp_mse_x2[k] = np.mean((FEt_mlp_b2[k] - b2True)**2)  # MSE (df adjusted)
    FEt_mlp_mae_x2[k] = np.mean(np.abs(FEt_mlp_b2[k] - b2True))  # MAE (df adjusted)
    FEt_mlp_mpe_x2[k] = np.mean((b2True - FEt_mlp_b2[k]) / b2True)  # MPE (df adjusted)
    FEt_mlp_mape_x2[k] = np.abs(FEt_mlp_mpe_x2[k])
    FEt_mlp_mse_x3[k] = np.mean((FEt_mlp_b3[k] - b3True)**3)  # MSE (df adjusted)
    FEt_mlp_mae_x3[k] = np.mean(np.abs(FEt_mlp_b3[k] - b3True))  # MAE (df adjusted)
    FEt_mlp_mpe_x3[k] = np.mean((b3True - FEt_mlp_b3[k]) / b3True)  # MPE (df adjusted)
    FEt_mlp_mape_x3[k] = np.abs(FEt_mlp_mpe_x3[k])
    
    FEt_mlp_sig2[k] = FEt_mlp_.scale
    
    FEt_mlp_b0SE[k] = FEt_mlp_.bse['Intercept']
    FEt_mlp_b1SE[k] = FEt_mlp_.bse['x_1']
    FEt_mlp_b2SE[k] = FEt_mlp_.bse['x_2']
    FEt_mlp_b3SE[k] = FEt_mlp_.bse['x_3']
    
    CIlo_FEt_mlp_b0[k] = FEt_mlp_b0[k] - (1.96 * FEt_mlp_b0SE[k])
    CIhi_FEt_mlp_b0[k] = FEt_mlp_b0[k] + (1.96 * FEt_mlp_b0SE[k])
    CIlo_FEt_mlp_b1[k] = FEt_mlp_b1[k] - (1.96 * FEt_mlp_b1SE[k])
    CIhi_FEt_mlp_b1[k] = FEt_mlp_b1[k] + (1.96 * FEt_mlp_b1SE[k])
    CIlo_FEt_mlp_b2[k] = FEt_mlp_b2[k] - (1.96 * FEt_mlp_b2SE[k])
    CIhi_FEt_mlp_b2[k] = FEt_mlp_b2[k] + (1.96 * FEt_mlp_b2SE[k])
    CIlo_FEt_mlp_b3[k] = FEt_mlp_b3[k] - (1.96 * FEt_mlp_b3SE[k])
    CIhi_FEt_mlp_b3[k] = FEt_mlp_b3[k] + (1.96 * FEt_mlp_b3SE[k])
    
    p = len(FEt_mlp_.params)
    obs = len(FEt_mlp_.resid)
    FEt_mlp_aic[k] = 2 * p + obs * (np.log(2 * np.pi) + FEt_mlp_.scale)

    FEt_mlp_b1Var[k] = (FEt_mlp_b1SE[k])**2
    FEt_mlp_b2Var[k] = (FEt_mlp_b2SE[k])**2
    FEt_mlp_b3Var[k] = (FEt_mlp_b3SE[k])**2

    ### Random Effects (Time Level) ###
    # Prepare data for panel regression
    X_mlp = X_mlp.rename(columns={'x.1': 'x_1', 'x.2': 'x_2', 'x.3': 'x_3'})
    #X_mlp = X_mlp.assign(y=y)
    #X_mlp = X_mlp.set_index('paper')
    #X_mlp['y'] = y
    
    #X_mlp['t'] = X_mlp.groupby('year').cumcount()
    #X_mlp = X_mlp.set_index(['country', 't'])
    #X_mi = X_mi.set_index(['country', 'year'])
    X_mlp = X_mlp.set_index(['paper','year'])      #time level
    
    # Fit mixed linear model with random intercept grouped by paper
    RE_mlp_ = RandomEffects.from_formula('y ~ 1 + x_1 + x_2 + x_3',data=X_mlp).fit()
    
    # Calculate (adjusted) R-squared
    RE_mlp_r2[k] = RE_mlp_.rsquared
    RE_mlp_a_r2[k] = 1 - (1 - RE_mlp_.rsquared) * (n - 1) / (n - p - 1)
    
    RE_mlp_mse[k] = np.mean(RE_mlp_.resids.values**2)  # MSE (df adjusted)
    RE_mlp_mae[k] = np.mean(np.abs(RE_mlp_.resids.values))  # MAE (df adjusted)
    RE_mlp_mpe[k] = np.mean(RE_mlp_.resids.values / X_mlp['y'].values)  # MPE (df adjusted)
    RE_mlp_mape[k] = np.abs(RE_mlp_mpe[k])
    
    RE_mlp_b0[k] = RE_mlp_.params['Intercept']
    RE_mlp_b1[k] = RE_mlp_.params['x_1']
    RE_mlp_b2[k] = RE_mlp_.params['x_2']
    RE_mlp_b3[k] = RE_mlp_.params['x_3']
    
    RE_mlp_mse_x1[k] = np.mean((RE_mlp_b1[k] - b1True)**2)  # MSE (df adjusted)
    RE_mlp_mae_x1[k] = np.mean(np.abs(RE_mlp_b1[k] - b1True))  # MAE (df adjusted)
    RE_mlp_mpe_x1[k] = np.mean((b1True - RE_mlp_b1[k]) / b1True)  # MPE (df adjusted)
    RE_mlp_mape_x1[k] = np.abs(RE_mlp_mpe_x1[k])
    RE_mlp_mse_x2[k] = np.mean((RE_mlp_b2[k] - b2True)**2)  # MSE (df adjusted)
    RE_mlp_mae_x2[k] = np.mean(np.abs(RE_mlp_b2[k] - b2True))  # MAE (df adjusted)
    RE_mlp_mpe_x2[k] = np.mean((b2True - RE_mlp_b2[k]) / b2True)  # MPE (df adjusted)
    RE_mlp_mape_x2[k] = np.abs(RE_mlp_mpe_x2[k])
    RE_mlp_mse_x3[k] = np.mean((RE_mlp_b3[k] - b3True)**2)  # MSE (df adjusted)
    RE_mlp_mae_x3[k] = np.mean(np.abs(RE_mlp_b3[k] - b3True))  # MAE (df adjusted)
    RE_mlp_mpe_x3[k] = np.mean((b3True - RE_mlp_b3[k]) / b3True)  # MPE (df adjusted)
    RE_mlp_mape_x3[k] = np.abs(RE_mlp_mpe_x3[k])
    
    RE_mlp_sig2[k] = RE_mlp_.resids.var()  # within variance, like FE residual variance
    
    RE_mlp_b0SE[k] = RE_mlp_.std_errors['Intercept']
    RE_mlp_b1SE[k] = RE_mlp_.std_errors['x_1']
    RE_mlp_b2SE[k] = RE_mlp_.std_errors['x_2']
    RE_mlp_b3SE[k] = RE_mlp_.std_errors['x_3']
    
    CIlo_RE_mlp_b0[k] = RE_mlp_b0[k] - (1.96 * RE_mlp_b0SE[k])
    CIhi_RE_mlp_b0[k] = RE_mlp_b0[k] + (1.96 * RE_mlp_b0SE[k])
    CIlo_RE_mlp_b1[k] = RE_mlp_b1[k] - (1.96 * RE_mlp_b1SE[k])
    CIhi_RE_mlp_b1[k] = RE_mlp_b1[k] + (1.96 * RE_mlp_b1SE[k])
    CIlo_RE_mlp_b2[k] = RE_mlp_b2[k] - (1.96 * RE_mlp_b2SE[k])
    CIhi_RE_mlp_b2[k] = RE_mlp_b2[k] + (1.96 * RE_mlp_b2SE[k])
    CIlo_RE_mlp_b3[k] = RE_mlp_b3[k] - (1.96 * RE_mlp_b3SE[k])
    CIhi_RE_mlp_b3[k] = RE_mlp_b3[k] + (1.96 * RE_mlp_b3SE[k])
    
    p = len(RE_mlp_.params)
    obs = RE_mlp_.nobs
    sigma2 = RE_mlp_.resids.var()

    RE_mlp_aic[k] = 2 * p + obs * (np.log(2 * np.pi) + np.log(sigma2) + 1)

    RE_mlp_b1Var[k] = (RE_mlp_b1SE[k])**2
    RE_mlp_b2Var[k] = (RE_mlp_b2SE[k])**2
    RE_mlp_b3Var[k] = (RE_mlp_b3SE[k])**2



    ##### vae #####
    
    ### Study-level Fixed Effects ###
    FE_vae_ = smf.ols('y ~ x_1 + x_2 + x_3 + C(paper)', data=X_vae.rename(columns={'x.1': 'x_1', 'x.2': 'x_2', 'x.3': 'x_3'})).fit()  # fixed model
    FE_vae_result = FE_vae_.summary()

    FE_vae_r2[k] = FE_vae_.rsquared
    FE_vae_a_r2[k] = FE_vae_.rsquared_adj
    FE_vae_mse[k] = np.mean(FE_vae_.resid**2)  # MSE (df adjusted)
    FE_vae_mae[k] = np.mean(np.abs(FE_vae_.resid))  # MAE (df adjusted)
    FE_vae_mpe[k] = np.mean(FE_vae_.resid / X_vae['y'].values)  # MPE (df adjusted)
    FE_vae_mape[k] = np.abs(FE_vae_mpe[k])
    
    FE_vae_b0[k] = FE_vae_.params['Intercept']
    FE_vae_b1[k] = FE_vae_.params['x_1']
    FE_vae_b2[k] = FE_vae_.params['x_2']
    FE_vae_b3[k] = FE_vae_.params['x_3']
    
    FE_vae_mse_x1[k] = np.mean((FE_vae_b1[k] - b1True)**2)  # MSE (df adjusted)
    FE_vae_mae_x1[k] = np.mean(np.abs(FE_vae_b1[k] - b1True))  # MAE (df adjusted)
    FE_vae_mpe_x1[k] = np.mean((b1True - FE_vae_b1[k]) / b1True)  # MPE (df adjusted)
    FE_vae_mape_x1[k] = np.abs(FE_vae_mpe_x1[k])
    FE_vae_mse_x2[k] = np.mean((FE_vae_b2[k] - b2True)**2)  # MSE (df adjusted)
    FE_vae_mae_x2[k] = np.mean(np.abs(FE_vae_b2[k] - b2True))  # MAE (df adjusted)
    FE_vae_mpe_x2[k] = np.mean((b2True - FE_vae_b2[k]) / b2True)  # MPE (df adjusted)
    FE_vae_mape_x2[k] = np.abs(FE_vae_mpe_x2[k])
    FE_vae_mse_x3[k] = np.mean((FE_vae_b3[k] - b3True)**3)  # MSE (df adjusted)
    FE_vae_mae_x3[k] = np.mean(np.abs(FE_vae_b3[k] - b3True))  # MAE (df adjusted)
    FE_vae_mpe_x3[k] = np.mean((b3True - FE_vae_b3[k]) / b3True)  # MPE (df adjusted)
    FE_vae_mape_x3[k] = np.abs(FE_vae_mpe_x3[k])
    
    FE_vae_sig2[k] = FE_vae_.scale
    
    FE_vae_b0SE[k] = FE_vae_.bse['Intercept']
    FE_vae_b1SE[k] = FE_vae_.bse['x_1']
    FE_vae_b2SE[k] = FE_vae_.bse['x_2']
    FE_vae_b3SE[k] = FE_vae_.bse['x_3']
    
    CIlo_FE_vae_b0[k] = FE_vae_b0[k] - (1.96 * FE_vae_b0SE[k])
    CIhi_FE_vae_b0[k] = FE_vae_b0[k] + (1.96 * FE_vae_b0SE[k])
    CIlo_FE_vae_b1[k] = FE_vae_b1[k] - (1.96 * FE_vae_b1SE[k])
    CIhi_FE_vae_b1[k] = FE_vae_b1[k] + (1.96 * FE_vae_b1SE[k])
    CIlo_FE_vae_b2[k] = FE_vae_b2[k] - (1.96 * FE_vae_b2SE[k])
    CIhi_FE_vae_b2[k] = FE_vae_b2[k] + (1.96 * FE_vae_b2SE[k])
    CIlo_FE_vae_b3[k] = FE_vae_b3[k] - (1.96 * FE_vae_b3SE[k])
    CIhi_FE_vae_b3[k] = FE_vae_b3[k] + (1.96 * FE_vae_b3SE[k])
    
    p = len(FE_vae_.params)
    obs = len(FE_vae_.resid)
    FE_vae_aic[k] = 2 * p + obs * (np.log(2 * np.pi) + FE_vae_.scale)
    
    FE_vae_b1Var[k] = (FE_vae_b1SE[k])**2
    FE_vae_b2Var[k] = (FE_vae_b2SE[k])**2
    FE_vae_b3Var[k] = (FE_vae_b3SE[k])**2
    
    ### Time-level Fixed Effects ###
    FEt_vae_ = smf.ols('y ~ x_1 + x_2 + x_3 + C(year)', data=X_vae.rename(columns={'x.1': 'x_1', 'x.2': 'x_2', 'x.3': 'x_3'})).fit()  # fixed model
    FEt_vae_result = FEt_vae_.summary()
    
    FEt_vae_r2[k] = FEt_vae_.rsquared
    FEt_vae_a_r2[k] = FEt_vae_.rsquared_adj
    FEt_vae_mse[k] = np.mean(FEt_vae_.resid**2)  # MSE (df adjusted)
    FEt_vae_mae[k] = np.mean(np.abs(FEt_vae_.resid))  # MAE (df adjusted)
    FEt_vae_mpe[k] = np.mean(FEt_vae_.resid / X_vae['y'].values)  # MPE (df adjusted)
    FEt_vae_mape[k] = np.abs(FEt_vae_mpe[k])
    
    FEt_vae_b0[k] = FEt_vae_.params['Intercept']
    FEt_vae_b1[k] = FEt_vae_.params['x_1']
    FEt_vae_b2[k] = FEt_vae_.params['x_2']
    FEt_vae_b3[k] = FEt_vae_.params['x_3']
    
    FEt_vae_mse_x1[k] = np.mean((FEt_vae_b1[k] - b1True)**2)  # MSE (df adjusted)
    FEt_vae_mae_x1[k] = np.mean(np.abs(FEt_vae_b1[k] - b1True))  # MAE (df adjusted)
    FEt_vae_mpe_x1[k] = np.mean((b1True - FEt_vae_b1[k]) / b1True)  # MPE (df adjusted)
    FEt_vae_mape_x1[k] = np.abs(FEt_vae_mpe_x1[k])
    FEt_vae_mse_x2[k] = np.mean((FEt_vae_b2[k] - b2True)**2)  # MSE (df adjusted)
    FEt_vae_mae_x2[k] = np.mean(np.abs(FEt_vae_b2[k] - b2True))  # MAE (df adjusted)
    FEt_vae_mpe_x2[k] = np.mean((b2True - FEt_vae_b2[k]) / b2True)  # MPE (df adjusted)
    FEt_vae_mape_x2[k] = np.abs(FEt_vae_mpe_x2[k])
    FEt_vae_mse_x3[k] = np.mean((FEt_vae_b3[k] - b3True)**3)  # MSE (df adjusted)
    FEt_vae_mae_x3[k] = np.mean(np.abs(FEt_vae_b3[k] - b3True))  # MAE (df adjusted)
    FEt_vae_mpe_x3[k] = np.mean((b3True - FEt_vae_b3[k]) / b3True)  # MPE (df adjusted)
    FEt_vae_mape_x3[k] = np.abs(FEt_vae_mpe_x3[k])
    
    FEt_vae_sig2[k] = FEt_vae_.scale
    
    FEt_vae_b0SE[k] = FEt_vae_.bse['Intercept']
    FEt_vae_b1SE[k] = FEt_vae_.bse['x_1']
    FEt_vae_b2SE[k] = FEt_vae_.bse['x_2']
    FEt_vae_b3SE[k] = FEt_vae_.bse['x_3']
    
    CIlo_FEt_vae_b0[k] = FEt_vae_b0[k] - (1.96 * FEt_vae_b0SE[k])
    CIhi_FEt_vae_b0[k] = FEt_vae_b0[k] + (1.96 * FEt_vae_b0SE[k])
    CIlo_FEt_vae_b1[k] = FEt_vae_b1[k] - (1.96 * FEt_vae_b1SE[k])
    CIhi_FEt_vae_b1[k] = FEt_vae_b1[k] + (1.96 * FEt_vae_b1SE[k])
    CIlo_FEt_vae_b2[k] = FEt_vae_b2[k] - (1.96 * FEt_vae_b2SE[k])
    CIhi_FEt_vae_b2[k] = FEt_vae_b2[k] + (1.96 * FEt_vae_b2SE[k])
    CIlo_FEt_vae_b3[k] = FEt_vae_b3[k] - (1.96 * FEt_vae_b3SE[k])
    CIhi_FEt_vae_b3[k] = FEt_vae_b3[k] + (1.96 * FEt_vae_b3SE[k])
    
    p = len(FEt_vae_.params)
    obs = len(FEt_vae_.resid)
    FEt_vae_aic[k] = 2 * p + obs * (np.log(2 * np.pi) + FEt_vae_.scale)

    FEt_vae_b1Var[k] = (FEt_vae_b1SE[k])**2
    FEt_vae_b2Var[k] = (FEt_vae_b2SE[k])**2
    FEt_vae_b3Var[k] = (FEt_vae_b3SE[k])**2

    ### Random Effects (Time Level) ###
    # Prepare data for panel regression
    X_vae = X_vae.rename(columns={'x.1': 'x_1', 'x.2': 'x_2', 'x.3': 'x_3'})
    #X_vae = X_vae.assign(y=y)
    #X_vae = X_vae.set_index('paper')
    #X_vae['y'] = y
    
    X_vae['t'] = X_vae.groupby('year').cumcount()
    X_vae = X_vae.set_index(['country', 't'])
    #X_mi = X_mi.set_index(['country', 'year'])
    X_vae = X_vae.set_index(['paper','year'])      #time level
    
    # Fit mixed linear model with random intercept grouped by paper
    RE_vae_ = RandomEffects.from_formula('y ~ 1 + x_1 + x_2 + x_3',data=X_vae).fit()
    
    # Calculate (adjusted) R-squared
    RE_vae_r2[k] = RE_vae_.rsquared
    RE_vae_a_r2[k] = 1 - (1 - RE_vae_.rsquared) * (n - 1) / (n - p - 1)
    
    RE_vae_mse[k] = np.mean(RE_vae_.resids.values**2)  # MSE (df adjusted)
    RE_vae_mae[k] = np.mean(np.abs(RE_vae_.resids.values))  # MAE (df adjusted)
    RE_vae_mpe[k] = np.mean(RE_vae_.resids.values / X_vae['y'].values)  # MPE (df adjusted)
    RE_vae_mape[k] = np.abs(RE_vae_mpe[k])
    
    RE_vae_b0[k] = RE_vae_.params['Intercept']
    RE_vae_b1[k] = RE_vae_.params['x_1']
    RE_vae_b2[k] = RE_vae_.params['x_2']
    RE_vae_b3[k] = RE_vae_.params['x_3']
    
    RE_vae_mse_x1[k] = np.mean((RE_vae_b1[k] - b1True)**2)  # MSE (df adjusted)
    RE_vae_mae_x1[k] = np.mean(np.abs(RE_vae_b1[k] - b1True))  # MAE (df adjusted)
    RE_vae_mpe_x1[k] = np.mean((b1True - RE_vae_b1[k]) / b1True)  # MPE (df adjusted)
    RE_vae_mape_x1[k] = np.abs(RE_vae_mpe_x1[k])
    RE_vae_mse_x2[k] = np.mean((RE_vae_b2[k] - b2True)**2)  # MSE (df adjusted)
    RE_vae_mae_x2[k] = np.mean(np.abs(RE_vae_b2[k] - b2True))  # MAE (df adjusted)
    RE_vae_mpe_x2[k] = np.mean((b2True - RE_vae_b2[k]) / b2True)  # MPE (df adjusted)
    RE_vae_mape_x2[k] = np.abs(RE_vae_mpe_x2[k])
    RE_vae_mse_x3[k] = np.mean((RE_vae_b3[k] - b3True)**2)  # MSE (df adjusted)
    RE_vae_mae_x3[k] = np.mean(np.abs(RE_vae_b3[k] - b3True))  # MAE (df adjusted)
    RE_vae_mpe_x3[k] = np.mean((b3True - RE_vae_b3[k]) / b3True)  # MPE (df adjusted)
    RE_vae_mape_x3[k] = np.abs(RE_vae_mpe_x3[k])
    
    RE_vae_sig2[k] = RE_vae_.resids.var()  # within variance, like FE residual variance
    
    RE_vae_b0SE[k] = RE_vae_.std_errors['Intercept']
    RE_vae_b1SE[k] = RE_vae_.std_errors['x_1']
    RE_vae_b2SE[k] = RE_vae_.std_errors['x_2']
    RE_vae_b3SE[k] = RE_vae_.std_errors['x_3']
    
    CIlo_RE_vae_b0[k] = RE_vae_b0[k] - (1.96 * RE_vae_b0SE[k])
    CIhi_RE_vae_b0[k] = RE_vae_b0[k] + (1.96 * RE_vae_b0SE[k])
    CIlo_RE_vae_b1[k] = RE_vae_b1[k] - (1.96 * RE_vae_b1SE[k])
    CIhi_RE_vae_b1[k] = RE_vae_b1[k] + (1.96 * RE_vae_b1SE[k])
    CIlo_RE_vae_b2[k] = RE_vae_b2[k] - (1.96 * RE_vae_b2SE[k])
    CIhi_RE_vae_b2[k] = RE_vae_b2[k] + (1.96 * RE_vae_b2SE[k])
    CIlo_RE_vae_b3[k] = RE_vae_b3[k] - (1.96 * RE_vae_b3SE[k])
    CIhi_RE_vae_b3[k] = RE_vae_b3[k] + (1.96 * RE_vae_b3SE[k])
    
    p = len(RE_vae_.params)
    obs = RE_vae_.nobs
    sigma2 = RE_vae_.resids.var()

    RE_vae_aic[k] = 2 * p + obs * (np.log(2 * np.pi) + np.log(sigma2) + 1)

    RE_vae_b1Var[k] = (RE_vae_b1SE[k])**2
    RE_vae_b2Var[k] = (RE_vae_b2SE[k])**2
    RE_vae_b3Var[k] = (RE_vae_b3SE[k])**2
    
    

    ##### gae #####
    
    ### Study-level Fixed Effects ###    
    FE_gae_ = smf.ols('y ~ x_1 + x_2 + x_3 + C(paper)', data=X_gae.rename(columns={'x.1': 'x_1', 'x.2': 'x_2', 'x.3': 'x_3'})).fit()  # fixed model
    #
    FE_gae_params, FE_gae_ses = arc_safe_regression(FE_gae_)
    FE_gae_result = FE_gae_.summary()

    FE_gae_r2[k] = FE_gae_.rsquared
    FE_gae_a_r2[k] = FE_gae_.rsquared_adj
    FE_gae_mse[k] = np.mean(FE_gae_.resid**2)  # MSE (df adjusted)
    FE_gae_mae[k] = np.mean(np.abs(FE_gae_.resid))  # MAE (df adjusted)
    FE_gae_mpe[k] = np.mean(FE_gae_.resid / X_gae['y'].values)  # MPE (df adjusted)
    FE_gae_mape[k] = np.abs(FE_gae_mpe[k])
    
    FE_gae_b0[k] = FE_gae_params['Intercept']
    FE_gae_b1[k] = FE_gae_params['x_1']
    FE_gae_b2[k] = FE_gae_params['x_2']
    FE_gae_b3[k] = FE_gae_params['x_3']
    
    FE_gae_mse_x1[k] = np.mean((FE_gae_b1[k] - b1True)**2)  # MSE (df adjusted)
    FE_gae_mae_x1[k] = np.mean(np.abs(FE_gae_b1[k] - b1True))  # MAE (df adjusted)
    FE_gae_mpe_x1[k] = np.mean((b1True - FE_gae_b1[k]) / b1True)  # MPE (df adjusted)
    FE_gae_mape_x1[k] = np.abs(FE_gae_mpe_x1[k])
    FE_gae_mse_x2[k] = np.mean((FE_gae_b2[k] - b2True)**2)  # MSE (df adjusted)
    FE_gae_mae_x2[k] = np.mean(np.abs(FE_gae_b2[k] - b2True))  # MAE (df adjusted)
    FE_gae_mpe_x2[k] = np.mean((b2True - FE_gae_b2[k]) / b2True)  # MPE (df adjusted)
    FE_gae_mape_x2[k] = np.abs(FE_gae_mpe_x2[k])
    FE_gae_mse_x3[k] = np.mean((FE_gae_b3[k] - b3True)**3)  # MSE (df adjusted)
    FE_gae_mae_x3[k] = np.mean(np.abs(FE_gae_b3[k] - b3True))  # MAE (df adjusted)
    FE_gae_mpe_x3[k] = np.mean((b3True - FE_gae_b3[k]) / b3True)  # MPE (df adjusted)
    FE_gae_mape_x3[k] = np.abs(FE_gae_mpe_x3[k])
    
    FE_gae_sig2[k] = FE_gae_.scale
    
    FE_gae_b0SE[k] = FE_gae_ses['Intercept']
    FE_gae_b1SE[k] = FE_gae_ses['x_1']
    FE_gae_b2SE[k] = FE_gae_ses['x_2']
    FE_gae_b3SE[k] = FE_gae_ses['x_3']
        
    CIlo_FE_gae_b0[k] = FE_gae_b0[k] - (1.96 * FE_gae_b0SE[k])
    CIhi_FE_gae_b0[k] = FE_gae_b0[k] + (1.96 * FE_gae_b0SE[k])
    CIlo_FE_gae_b1[k] = FE_gae_b1[k] - (1.96 * FE_gae_b1SE[k])
    CIhi_FE_gae_b1[k] = FE_gae_b1[k] + (1.96 * FE_gae_b1SE[k])
    CIlo_FE_gae_b2[k] = FE_gae_b2[k] - (1.96 * FE_gae_b2SE[k])
    CIhi_FE_gae_b2[k] = FE_gae_b2[k] + (1.96 * FE_gae_b2SE[k])
    CIlo_FE_gae_b3[k] = FE_gae_b3[k] - (1.96 * FE_gae_b3SE[k])
    CIhi_FE_gae_b3[k] = FE_gae_b3[k] + (1.96 * FE_gae_b3SE[k])
    
    p = len(FE_gae_.params)
    obs = len(FE_gae_.resid)
    FE_gae_aic[k] = 2 * p + obs * (np.log(2 * np.pi) + FE_gae_.scale)
    
    FE_gae_b1Var[k] = (FE_gae_b1SE[k])**2
    FE_gae_b2Var[k] = (FE_gae_b2SE[k])**2
    FE_gae_b3Var[k] = (FE_gae_b3SE[k])**2
    
    ### Time-level Fixed Effects ###
    FEt_gae_ = smf.ols('y ~ x_1 + x_2 + x_3 + C(year)', data=X_gae.rename(columns={'x.1': 'x_1', 'x.2': 'x_2', 'x.3': 'x_3'})).fit()  # fixed model
    #
    FEt_gae_params, FEt_gae_ses = arc_safe_regression(FEt_gae_)
    FEt_gae_result = FEt_gae_.summary()
    
    FEt_gae_r2[k] = FEt_gae_.rsquared
    FEt_gae_a_r2[k] = FEt_gae_.rsquared_adj
    FEt_gae_mse[k] = np.mean(FEt_gae_.resid**2)  # MSE (df adjusted)
    FEt_gae_mae[k] = np.mean(np.abs(FEt_gae_.resid))  # MAE (df adjusted)
    FEt_gae_mpe[k] = np.mean(FEt_gae_.resid / X_gae['y'].values)  # MPE (df adjusted)
    FEt_gae_mape[k] = np.abs(FEt_gae_mpe[k])
    
    FEt_gae_b0[k] = FEt_gae_params['Intercept']
    FEt_gae_b1[k] = FEt_gae_params['x_1']
    FEt_gae_b2[k] = FEt_gae_params['x_2']
    FEt_gae_b3[k] = FEt_gae_params['x_3']
    
    FEt_gae_mse_x1[k] = np.mean((FEt_gae_b1[k] - b1True)**2)  # MSE (df adjusted)
    FEt_gae_mae_x1[k] = np.mean(np.abs(FEt_gae_b1[k] - b1True))  # MAE (df adjusted)
    FEt_gae_mpe_x1[k] = np.mean((b1True - FEt_gae_b1[k]) / b1True)  # MPE (df adjusted)
    FEt_gae_mape_x1[k] = np.abs(FEt_gae_mpe_x1[k])
    FEt_gae_mse_x2[k] = np.mean((FEt_gae_b2[k] - b2True)**2)  # MSE (df adjusted)
    FEt_gae_mae_x2[k] = np.mean(np.abs(FEt_gae_b2[k] - b2True))  # MAE (df adjusted)
    FEt_gae_mpe_x2[k] = np.mean((b2True - FEt_gae_b2[k]) / b2True)  # MPE (df adjusted)
    FEt_gae_mape_x2[k] = np.abs(FEt_gae_mpe_x2[k])
    FEt_gae_mse_x3[k] = np.mean((FEt_gae_b3[k] - b3True)**3)  # MSE (df adjusted)
    FEt_gae_mae_x3[k] = np.mean(np.abs(FEt_gae_b3[k] - b3True))  # MAE (df adjusted)
    FEt_gae_mpe_x3[k] = np.mean((b3True - FEt_gae_b3[k]) / b3True)  # MPE (df adjusted)
    FEt_gae_mape_x3[k] = np.abs(FEt_gae_mpe_x3[k])
    
    FEt_gae_sig2[k] = FEt_gae_.scale
    
    FEt_gae_b0SE[k] = FEt_gae_ses['Intercept']
    FEt_gae_b1SE[k] = FEt_gae_ses['x_1']
    FEt_gae_b2SE[k] = FEt_gae_ses['x_2']
    FEt_gae_b3SE[k] = FEt_gae_ses['x_3']
    
    CIlo_FEt_gae_b0[k] = FEt_gae_b0[k] - (1.96 * FEt_gae_b0SE[k])
    CIhi_FEt_gae_b0[k] = FEt_gae_b0[k] + (1.96 * FEt_gae_b0SE[k])
    CIlo_FEt_gae_b1[k] = FEt_gae_b1[k] - (1.96 * FEt_gae_b1SE[k])
    CIhi_FEt_gae_b1[k] = FEt_gae_b1[k] + (1.96 * FEt_gae_b1SE[k])
    CIlo_FEt_gae_b2[k] = FEt_gae_b2[k] - (1.96 * FEt_gae_b2SE[k])
    CIhi_FEt_gae_b2[k] = FEt_gae_b2[k] + (1.96 * FEt_gae_b2SE[k])
    CIlo_FEt_gae_b3[k] = FEt_gae_b3[k] - (1.96 * FEt_gae_b3SE[k])
    CIhi_FEt_gae_b3[k] = FEt_gae_b3[k] + (1.96 * FEt_gae_b3SE[k])
    
    p = len(FEt_gae_.params)
    obs = len(FEt_gae_.resid)
    FEt_gae_aic[k] = 2 * p + obs * (np.log(2 * np.pi) + FEt_gae_.scale)

    FEt_gae_b1Var[k] = (FEt_gae_b1SE[k])**2
    FEt_gae_b2Var[k] = (FEt_gae_b2SE[k])**2
    FEt_gae_b3Var[k] = (FEt_gae_b3SE[k])**2

    ### Random Effects (Time Level) ###
    
    # HPC rounds y's variation down to zero, so can't compile
    
    # Prepare data for panel regression
    X_gae = X_gae.rename(columns={'x.1': 'x_1', 'x.2': 'x_2', 'x.3': 'x_3'})
    #X_gae = X_gae.assign(y=y)
    #X_gae = X_gae.set_index('paper')
    #X_gae['y'] = y
    
    X_gae['t'] = X_gae.groupby('year').cumcount()
    X_gae = X_gae.set_index(['country', 't'])
    #X_mi = X_mi.set_index(['country', 'year'])
    X_gae = X_gae.set_index(['paper','year'])      #time level
    
    # Fit mixed linear model with random intercept grouped by paper
    RE_gae_ = RandomEffects.from_formula('y ~ 1 + x_1 + x_2 + x_3',data=X_gae).fit()
    
    # Calculate (adjusted) R-squared
    RE_gae_r2[k] = RE_gae_.rsquared
    RE_gae_a_r2[k] = 1 - (1 - RE_gae_.rsquared) * (n - 1) / (n - p - 1)
    
    RE_gae_mse[k] = np.mean(RE_gae_.resids.values**2)  # MSE (df adjusted)
    RE_gae_mae[k] = np.mean(np.abs(RE_gae_.resids.values))  # MAE (df adjusted)
    RE_gae_mpe[k] = np.mean(RE_gae_.resids.values / X_gae['y'].values)  # MPE (df adjusted)
    RE_gae_mape[k] = np.abs(RE_gae_mpe[k])
    
    RE_gae_b0[k] = RE_gae_.params['Intercept']
    RE_gae_b1[k] = RE_gae_.params['x_1']
    RE_gae_b2[k] = RE_gae_.params['x_2']
    RE_gae_b3[k] = RE_gae_.params['x_3']
    
    RE_gae_mse_x1[k] = np.mean((RE_gae_b1[k] - b1True)**2)  # MSE (df adjusted)
    RE_gae_mae_x1[k] = np.mean(np.abs(RE_gae_b1[k] - b1True))  # MAE (df adjusted)
    RE_gae_mpe_x1[k] = np.mean((b1True - RE_gae_b1[k]) / b1True)  # MPE (df adjusted)
    RE_gae_mape_x1[k] = np.abs(RE_gae_mpe_x1[k])
    RE_gae_mse_x2[k] = np.mean((RE_gae_b2[k] - b2True)**2)  # MSE (df adjusted)
    RE_gae_mae_x2[k] = np.mean(np.abs(RE_gae_b2[k] - b2True))  # MAE (df adjusted)
    RE_gae_mpe_x2[k] = np.mean((b2True - RE_gae_b2[k]) / b2True)  # MPE (df adjusted)
    RE_gae_mape_x2[k] = np.abs(RE_gae_mpe_x2[k])
    RE_gae_mse_x3[k] = np.mean((RE_gae_b3[k] - b3True)**2)  # MSE (df adjusted)
    RE_gae_mae_x3[k] = np.mean(np.abs(RE_gae_b3[k] - b3True))  # MAE (df adjusted)
    RE_gae_mpe_x3[k] = np.mean((b3True - RE_gae_b3[k]) / b3True)  # MPE (df adjusted)
    RE_gae_mape_x3[k] = np.abs(RE_gae_mpe_x3[k])
    
    RE_gae_sig2[k] = RE_gae_.resids.var()  # within variance, like FE residual variance
    
    RE_gae_b0SE[k] = RE_gae_.std_errors['Intercept']
    RE_gae_b1SE[k] = RE_gae_.std_errors['x_1']
    RE_gae_b2SE[k] = RE_gae_.std_errors['x_2']
    RE_gae_b3SE[k] = RE_gae_.std_errors['x_3']
    
    CIlo_RE_gae_b0[k] = RE_gae_b0[k] - (1.96 * RE_gae_b0SE[k])
    CIhi_RE_gae_b0[k] = RE_gae_b0[k] + (1.96 * RE_gae_b0SE[k])
    CIlo_RE_gae_b1[k] = RE_gae_b1[k] - (1.96 * RE_gae_b1SE[k])
    CIhi_RE_gae_b1[k] = RE_gae_b1[k] + (1.96 * RE_gae_b1SE[k])
    CIlo_RE_gae_b2[k] = RE_gae_b2[k] - (1.96 * RE_gae_b2SE[k])
    CIhi_RE_gae_b2[k] = RE_gae_b2[k] + (1.96 * RE_gae_b2SE[k])
    CIlo_RE_gae_b3[k] = RE_gae_b3[k] - (1.96 * RE_gae_b3SE[k])
    CIhi_RE_gae_b3[k] = RE_gae_b3[k] + (1.96 * RE_gae_b3SE[k])
    
    p = len(RE_gae_.params)
    obs = RE_gae_.nobs
    sigma2 = RE_gae_.resids.var()

    RE_gae_aic[k] = 2 * p + obs * (np.log(2 * np.pi) + np.log(sigma2) + 1)

    RE_gae_b1Var[k] = (RE_gae_b1SE[k])**2
    RE_gae_b2Var[k] = (RE_gae_b2SE[k])**2
    RE_gae_b3Var[k] = (RE_gae_b3SE[k])**2



    ##### dif #####
    
    ### Study-level Fixed Effects ###    
    FE_dif_ = smf.ols('y ~ x_1 + x_2 + x_3 + C(paper)', data=X_dif.rename(columns={'x.1': 'x_1', 'x.2': 'x_2', 'x.3': 'x_3'})).fit()  # fixed model
    #
    FE_dif_params, FE_dif_ses = arc_safe_regression(FE_dif_)
    FE_dif_result = FE_dif_.summary()

    FE_dif_r2[k] = FE_dif_.rsquared
    FE_dif_a_r2[k] = FE_dif_.rsquared_adj
    FE_dif_mse[k] = np.mean(FE_dif_.resid**2)  # MSE (df adjusted)
    FE_dif_mae[k] = np.mean(np.abs(FE_dif_.resid))  # MAE (df adjusted)
    FE_dif_mpe[k] = np.mean(FE_dif_.resid / X_dif['y'].values)  # MPE (df adjusted)
    FE_dif_mape[k] = np.abs(FE_dif_mpe[k])
    
    FE_dif_b0[k] = FE_dif_params['Intercept']
    FE_dif_b1[k] = FE_dif_params['x_1']
    FE_dif_b2[k] = FE_dif_params['x_2']
    FE_dif_b3[k] = FE_dif_params['x_3']
    
    FE_dif_mse_x1[k] = np.mean((FE_dif_b1[k] - b1True)**2)  # MSE (df adjusted)
    FE_dif_mae_x1[k] = np.mean(np.abs(FE_dif_b1[k] - b1True))  # MAE (df adjusted)
    FE_dif_mpe_x1[k] = np.mean((b1True - FE_dif_b1[k]) / b1True)  # MPE (df adjusted)
    FE_dif_mape_x1[k] = np.abs(FE_dif_mpe_x1[k])
    FE_dif_mse_x2[k] = np.mean((FE_dif_b2[k] - b2True)**2)  # MSE (df adjusted)
    FE_dif_mae_x2[k] = np.mean(np.abs(FE_dif_b2[k] - b2True))  # MAE (df adjusted)
    FE_dif_mpe_x2[k] = np.mean((b2True - FE_dif_b2[k]) / b2True)  # MPE (df adjusted)
    FE_dif_mape_x2[k] = np.abs(FE_dif_mpe_x2[k])
    FE_dif_mse_x3[k] = np.mean((FE_dif_b3[k] - b3True)**3)  # MSE (df adjusted)
    FE_dif_mae_x3[k] = np.mean(np.abs(FE_dif_b3[k] - b3True))  # MAE (df adjusted)
    FE_dif_mpe_x3[k] = np.mean((b3True - FE_dif_b3[k]) / b3True)  # MPE (df adjusted)
    FE_dif_mape_x3[k] = np.abs(FE_dif_mpe_x3[k])
    
    FE_dif_sig2[k] = FE_dif_.scale
    
    FE_dif_b0SE[k] = FE_dif_ses['Intercept']
    FE_dif_b1SE[k] = FE_dif_ses['x_1']
    FE_dif_b2SE[k] = FE_dif_ses['x_2']
    FE_dif_b3SE[k] = FE_dif_ses['x_3']
    
    CIlo_FE_dif_b0[k] = FE_dif_b0[k] - (1.96 * FE_dif_b0SE[k])
    CIhi_FE_dif_b0[k] = FE_dif_b0[k] + (1.96 * FE_dif_b0SE[k])
    CIlo_FE_dif_b1[k] = FE_dif_b1[k] - (1.96 * FE_dif_b1SE[k])
    CIhi_FE_dif_b1[k] = FE_dif_b1[k] + (1.96 * FE_dif_b1SE[k])
    CIlo_FE_dif_b2[k] = FE_dif_b2[k] - (1.96 * FE_dif_b2SE[k])
    CIhi_FE_dif_b2[k] = FE_dif_b2[k] + (1.96 * FE_dif_b2SE[k])
    CIlo_FE_dif_b3[k] = FE_dif_b3[k] - (1.96 * FE_dif_b3SE[k])
    CIhi_FE_dif_b3[k] = FE_dif_b3[k] + (1.96 * FE_dif_b3SE[k])
    
    p = len(FE_dif_.params)
    obs = len(FE_dif_.resid)
    FE_dif_aic[k] = 2 * p + obs * (np.log(2 * np.pi) + FE_dif_.scale)
    
    FE_dif_b1Var[k] = (FE_dif_b1SE[k])**2
    FE_dif_b2Var[k] = (FE_dif_b2SE[k])**2
    FE_dif_b3Var[k] = (FE_dif_b3SE[k])**2


    
    ### Time-level Fixed Effects ###
    FEt_dif_ = smf.ols('y ~ x_1 + x_2 + x_3 + C(year)', data=X_dif.rename(columns={'x.1': 'x_1', 'x.2': 'x_2', 'x.3': 'x_3'})).fit()  # fixed model
    #
    FEt_dif_params, FEt_dif_ses = arc_safe_regression(FEt_dif_)
    FEt_dif_result = FEt_dif_.summary()
    
    FEt_dif_r2[k] = FEt_dif_.rsquared
    FEt_dif_a_r2[k] = FEt_dif_.rsquared_adj
    FEt_dif_mse[k] = np.mean(FEt_dif_.resid**2)  # MSE (df adjusted)
    FEt_dif_mae[k] = np.mean(np.abs(FEt_dif_.resid))  # MAE (df adjusted)
    FEt_dif_mpe[k] = np.mean(FEt_dif_.resid / X_dif['y'].values)  # MPE (df adjusted)
    FEt_dif_mape[k] = np.abs(FEt_dif_mpe[k])
    
    FEt_dif_b0[k] = FEt_dif_params['Intercept']
    FEt_dif_b1[k] = FEt_dif_params['x_1']
    FEt_dif_b2[k] = FEt_dif_params['x_2']
    FEt_dif_b3[k] = FEt_dif_params['x_3']
    
    FEt_dif_mse_x1[k] = np.mean((FEt_dif_b1[k] - b1True)**2)  # MSE (df adjusted)
    FEt_dif_mae_x1[k] = np.mean(np.abs(FEt_dif_b1[k] - b1True))  # MAE (df adjusted)
    FEt_dif_mpe_x1[k] = np.mean((b1True - FEt_dif_b1[k]) / b1True)  # MPE (df adjusted)
    FEt_dif_mape_x1[k] = np.abs(FEt_dif_mpe_x1[k])
    FEt_dif_mse_x2[k] = np.mean((FEt_dif_b2[k] - b2True)**2)  # MSE (df adjusted)
    FEt_dif_mae_x2[k] = np.mean(np.abs(FEt_dif_b2[k] - b2True))  # MAE (df adjusted)
    FEt_dif_mpe_x2[k] = np.mean((b2True - FEt_dif_b2[k]) / b2True)  # MPE (df adjusted)
    FEt_dif_mape_x2[k] = np.abs(FEt_dif_mpe_x2[k])
    FEt_dif_mse_x3[k] = np.mean((FEt_dif_b3[k] - b3True)**3)  # MSE (df adjusted)
    FEt_dif_mae_x3[k] = np.mean(np.abs(FEt_dif_b3[k] - b3True))  # MAE (df adjusted)
    FEt_dif_mpe_x3[k] = np.mean((b3True - FEt_dif_b3[k]) / b3True)  # MPE (df adjusted)
    FEt_dif_mape_x3[k] = np.abs(FEt_dif_mpe_x3[k])
    
    FEt_dif_sig2[k] = FEt_dif_.scale
    
    FEt_dif_b0SE[k] = FEt_dif_ses['Intercept']
    FEt_dif_b1SE[k] = FEt_dif_ses['x_1']
    FEt_dif_b2SE[k] = FEt_dif_ses['x_2']
    FEt_dif_b3SE[k] = FEt_dif_ses['x_3']
    
    CIlo_FEt_dif_b0[k] = FEt_dif_b0[k] - (1.96 * FEt_dif_b0SE[k])
    CIhi_FEt_dif_b0[k] = FEt_dif_b0[k] + (1.96 * FEt_dif_b0SE[k])
    CIlo_FEt_dif_b1[k] = FEt_dif_b1[k] - (1.96 * FEt_dif_b1SE[k])
    CIhi_FEt_dif_b1[k] = FEt_dif_b1[k] + (1.96 * FEt_dif_b1SE[k])
    CIlo_FEt_dif_b2[k] = FEt_dif_b2[k] - (1.96 * FEt_dif_b2SE[k])
    CIhi_FEt_dif_b2[k] = FEt_dif_b2[k] + (1.96 * FEt_dif_b2SE[k])
    CIlo_FEt_dif_b3[k] = FEt_dif_b3[k] - (1.96 * FEt_dif_b3SE[k])
    CIhi_FEt_dif_b3[k] = FEt_dif_b3[k] + (1.96 * FEt_dif_b3SE[k])
    
    p = len(FEt_dif_.params)
    obs = len(FEt_dif_.resid)
    FEt_dif_aic[k] = 2 * p + obs * (np.log(2 * np.pi) + FEt_dif_.scale)

    FEt_dif_b1Var[k] = (FEt_dif_b1SE[k])**2
    FEt_dif_b2Var[k] = (FEt_dif_b2SE[k])**2
    FEt_dif_b3Var[k] = (FEt_dif_b3SE[k])**2
    
    ### Random Effects (Time Level) ###
    # Prepare data for panel regression
    X_dif = X_dif.rename(columns={'x.1': 'x_1', 'x.2': 'x_2', 'x.3': 'x_3'})
    #X_dif = X_dif.assign(y=y)
    #X_dif = X_dif.set_index('paper')
    #X_dif['y'] = y
    
    X_dif['t'] = X_dif.groupby('year').cumcount()
    X_dif = X_dif.set_index(['country', 't'])
    #X_mi = X_mi.set_index(['country', 'year'])
    X_dif = X_dif.set_index(['paper','year'])      #time level
    
    # Fit mixed linear model with random intercept grouped by paper
    RE_dif_ = RandomEffects.from_formula('y ~ 1 + x_1 + x_2 + x_3',data=X_dif).fit()
    
    # Calculate (adjusted) R-squared
    RE_dif_r2[k] = RE_dif_.rsquared
    RE_dif_a_r2[k] = 1 - (1 - RE_dif_.rsquared) * (n - 1) / (n - p - 1)
    
    RE_dif_mse[k] = np.mean(RE_dif_.resids.values**2)  # MSE (df adjusted)
    RE_dif_mae[k] = np.mean(np.abs(RE_dif_.resids.values))  # MAE (df adjusted)
    RE_dif_mpe[k] = np.mean(RE_dif_.resids.values / X_dif['y'].values)  # MPE (df adjusted)
    RE_dif_mape[k] = np.abs(RE_dif_mpe[k])
    
    RE_dif_b0[k] = RE_dif_.params['Intercept']
    RE_dif_b1[k] = RE_dif_.params['x_1']
    RE_dif_b2[k] = RE_dif_.params['x_2']
    RE_dif_b3[k] = RE_dif_.params['x_3']
    
    RE_dif_mse_x1[k] = np.mean((RE_dif_b1[k] - b1True)**2)  # MSE (df adjusted)
    RE_dif_mae_x1[k] = np.mean(np.abs(RE_dif_b1[k] - b1True))  # MAE (df adjusted)
    RE_dif_mpe_x1[k] = np.mean((b1True - RE_dif_b1[k]) / b1True)  # MPE (df adjusted)
    RE_dif_mape_x1[k] = np.abs(RE_dif_mpe_x1[k])
    RE_dif_mse_x2[k] = np.mean((RE_dif_b2[k] - b2True)**2)  # MSE (df adjusted)
    RE_dif_mae_x2[k] = np.mean(np.abs(RE_dif_b2[k] - b2True))  # MAE (df adjusted)
    RE_dif_mpe_x2[k] = np.mean((b2True - RE_dif_b2[k]) / b2True)  # MPE (df adjusted)
    RE_dif_mape_x2[k] = np.abs(RE_dif_mpe_x2[k])
    RE_dif_mse_x3[k] = np.mean((RE_dif_b3[k] - b3True)**2)  # MSE (df adjusted)
    RE_dif_mae_x3[k] = np.mean(np.abs(RE_dif_b3[k] - b3True))  # MAE (df adjusted)
    RE_dif_mpe_x3[k] = np.mean((b3True - RE_dif_b3[k]) / b3True)  # MPE (df adjusted)
    RE_dif_mape_x3[k] = np.abs(RE_dif_mpe_x3[k])
    
    RE_dif_sig2[k] = RE_dif_.resids.var()  # within variance, like FE residual variance
    
    RE_dif_b0SE[k] = RE_dif_.std_errors['Intercept']
    RE_dif_b1SE[k] = RE_dif_.std_errors['x_1']
    RE_dif_b2SE[k] = RE_dif_.std_errors['x_2']
    RE_dif_b3SE[k] = RE_dif_.std_errors['x_3']
    
    CIlo_RE_dif_b0[k] = RE_dif_b0[k] - (1.96 * RE_dif_b0SE[k])
    CIhi_RE_dif_b0[k] = RE_dif_b0[k] + (1.96 * RE_dif_b0SE[k])
    CIlo_RE_dif_b1[k] = RE_dif_b1[k] - (1.96 * RE_dif_b1SE[k])
    CIhi_RE_dif_b1[k] = RE_dif_b1[k] + (1.96 * RE_dif_b1SE[k])
    CIlo_RE_dif_b2[k] = RE_dif_b2[k] - (1.96 * RE_dif_b2SE[k])
    CIhi_RE_dif_b2[k] = RE_dif_b2[k] + (1.96 * RE_dif_b2SE[k])
    CIlo_RE_dif_b3[k] = RE_dif_b3[k] - (1.96 * RE_dif_b3SE[k])
    CIhi_RE_dif_b3[k] = RE_dif_b3[k] + (1.96 * RE_dif_b3SE[k])
    
    p = len(RE_dif_.params)
    obs = RE_dif_.nobs
    sigma2 = RE_dif_.resids.var()
    
    RE_dif_aic[k] = 2 * p + obs * (np.log(2 * np.pi) + np.log(sigma2) + 1)
    
    RE_dif_b1Var[k] = (RE_dif_b1SE[k])**2
    RE_dif_b2Var[k] = (RE_dif_b2SE[k])**2
    RE_dif_b3Var[k] = (RE_dif_b3SE[k])**2

#chime.success()
#chime.info()

#chime.warning()
#chime.error()

#%%###########################################################################
###                             RESULTS - MONTE CARLO                      ###
##############################################################################

#filepath="G:/Other computers/My Laptop/SYNC/School/VT/RESEARCH/Dissertation/CH4 - ML/Code/"
filepath="/home/jegendron/Sim2/Case"+str(case)+"/"
#import matplotlib.pyplot as plt

###### FEs ######

### BIAS ###

fig, axs = plt.subplots(3, 3, figsize=(12, 16))
plt.subplots_adjust(hspace=0.4, wspace=0.4)
# Plot 1
axs[0, 0].hist(FE_cca_b0, color='tab:cyan', edgecolor='black')
#axs[0, 0].axvline(b0True, color='black', linewidth=3)
axs[0, 0].set_xlabel('CCA')
axs[0, 0].set_ylabel('Frequency')
# Plot 2
axs[0, 1].hist(FE_mi_b0, color='tab:blue', edgecolor='black')
#axs[0, 1].axvline(b0True, color='black', linewidth=3)
axs[0, 1].set_xlabel('MI')
axs[0, 1].set_ylabel('Frequency')
# Plot 3
axs[0, 2].hist(FE_lh_b0, color='tab:purple', edgecolor='black')
#axs[0, 2].axvline(b0True, color='black', linewidth=3)
axs[0, 2].set_xlabel('Likelihood')
axs[0, 2].set_ylabel('Frequency')
# Plot 4
axs[1, 0].hist(FE_rf_b0, color='tab:pink', edgecolor='black')
#axs[1, 0].axvline(b0True, color='black', linewidth=3)
axs[1, 0].set_xlabel('Random Forest')
axs[1, 0].set_ylabel('Frequency')
# Plot 5
axs[1, 1].hist(FE_lgb_b0, color='tab:orange', edgecolor='black')
#axs[1, 1].axvline(b0True, color='black', linewidth=3)
axs[1, 1].set_xlabel('LGBM')
axs[1, 1].set_ylabel('Frequency')
# Plot 5.5
axs[1, 2].hist(FE_mlp_b0, color='tab:red', edgecolor='black')
#axs[1, 2].axvline(b0True, color='black', linewidth=3)
axs[1, 2].set_xlabel('MLP')
axs[1, 2].set_ylabel('Frequency')
# Plot 6
axs[2, 0].hist(FE_vae_b0, color='tab:olive', edgecolor='black')
#axs[2, 0].axvline(b0True, color='black', linewidth=3)
axs[2, 0].set_xlabel('VAE')
axs[2, 0].set_ylabel('Frequency')
# Plot 7
axs[2, 1].hist(FE_gae_b0, color='tab:green', edgecolor='black')
#axs[2, 1].axvline(b0True, color='black', linewidth=3)
axs[2, 1].set_xlabel('GAE')
axs[2, 1].set_ylabel('Frequency')
# Plot 8
axs[2, 2].hist(FE_dif_b0, color='tab:brown', edgecolor='black')
#axs[2, 2].axvline(b0True, color='black', linewidth=3)
axs[2, 2].set_xlabel('Diffusion')
axs[2, 2].set_ylabel('Frequency')

fig.suptitle('Intercept Coefficient - Study Fixed Effects', fontsize=20)
#plt.savefig(filepath+"Figures/Case"+str(case)+"/BiasB0_FEs.png")
plt.savefig(filepath+"/BiasB0_FEs.png")
plt.close()

###

fig, axs = plt.subplots(3, 3, figsize=(12, 16))
plt.subplots_adjust(hspace=0.4, wspace=0.4)
# Plot 1
axs[0, 0].hist(FE_cca_b1, color='tab:cyan', edgecolor='black')
axs[0, 0].axvline(b1True, color='black', linewidth=3)
axs[0, 0].set_xlabel('CCA')
axs[0, 0].set_ylabel('Frequency')
# Plot 2
axs[0, 1].hist(FE_mi_b1, color='tab:blue', edgecolor='black')
axs[0, 1].axvline(b1True, color='black', linewidth=3)
axs[0, 1].set_xlabel('MI')
axs[0, 1].set_ylabel('Frequency')
# Plot 3
axs[0, 2].hist(FE_lh_b1, color='tab:purple', edgecolor='black')
axs[0, 2].axvline(b1True, color='black', linewidth=3)
axs[0, 2].set_xlabel('Likelihood')
axs[0, 2].set_ylabel('Frequency')
# Plot 4
axs[1, 0].hist(FE_rf_b1, color='tab:pink', edgecolor='black')
axs[1, 0].axvline(b1True, color='black', linewidth=3)
axs[1, 0].set_xlabel('Random Forest')
axs[1, 0].set_ylabel('Frequency')
# Plot 5
axs[1, 1].hist(FE_lgb_b1, color='tab:orange', edgecolor='black')
axs[1, 1].axvline(b1True, color='black', linewidth=3)
axs[1, 1].set_xlabel('LGBM')
axs[1, 1].set_ylabel('Frequency')
# Plot 5.5
axs[1, 2].hist(FE_mlp_b1, color='tab:red', edgecolor='black')
axs[1, 2].axvline(b1True, color='black', linewidth=3)
axs[1, 2].set_xlabel('MLP')
axs[1, 2].set_ylabel('Frequency')
# Plot 6
axs[2, 0].hist(FE_vae_b1, color='tab:olive', edgecolor='black')
axs[2, 0].axvline(b1True, color='black', linewidth=3)
axs[2, 0].set_xlabel('VAE')
axs[2, 0].set_ylabel('Frequency')
# Plot 7
axs[2, 1].hist(FE_gae_b1, color='tab:green', edgecolor='black')
axs[2, 1].axvline(b1True, color='black', linewidth=3)
axs[2, 1].set_xlabel('GAE')
axs[2, 1].set_ylabel('Frequency')
# Plot 8
axs[2, 2].hist(FE_dif_b1, color='tab:brown', edgecolor='black')
axs[2, 2].axvline(b1True, color='black', linewidth=3)
axs[2, 2].set_xlabel('Diffusion')
axs[2, 2].set_ylabel('Frequency')

fig.suptitle('x1 Coefficient - Study Fixed Effects', fontsize=20)
#plt.savefig(filepath+"Figures/Case"+str(case)+"/BiasB1_FEs.png")
plt.savefig(filepath+"/BiasB1_FEs.png")
plt.close()

###

fig, axs = plt.subplots(3, 3, figsize=(12, 16))
plt.subplots_adjust(hspace=0.4, wspace=0.4)
# Plot 1
axs[0, 0].hist(FE_cca_b2, color='tab:cyan', edgecolor='black')
axs[0, 0].axvline(b2True, color='black', linewidth=3)
axs[0, 0].set_xlabel('CCA')
axs[0, 0].set_ylabel('Frequency')
# Plot 2
axs[0, 1].hist(FE_mi_b2, color='tab:blue', edgecolor='black')
axs[0, 1].axvline(b2True, color='black', linewidth=3)
axs[0, 1].set_xlabel('MI')
axs[0, 1].set_ylabel('Frequency')
# Plot 3
axs[0, 2].hist(FE_lh_b2, color='tab:purple', edgecolor='black')
axs[0, 2].axvline(b2True, color='black', linewidth=3)
axs[0, 2].set_xlabel('Likelihood')
axs[0, 2].set_ylabel('Frequency')
# Plot 4
axs[1, 0].hist(FE_rf_b2, color='tab:pink', edgecolor='black')
axs[1, 0].axvline(b2True, color='black', linewidth=3)
axs[1, 0].set_xlabel('Random Forest')
axs[1, 0].set_ylabel('Frequency')
# Plot 5
axs[1, 1].hist(FE_lgb_b2, color='tab:orange', edgecolor='black')
axs[1, 1].axvline(b2True, color='black', linewidth=3)
axs[1, 1].set_xlabel('LGBM')
axs[1, 1].set_ylabel('Frequency')
# Plot 5.5
axs[1, 2].hist(FE_mlp_b2, color='tab:red', edgecolor='black')
axs[1, 2].axvline(b2True, color='black', linewidth=3)
axs[1, 2].set_xlabel('MLP')
axs[1, 2].set_ylabel('Frequency')
# Plot 6
axs[2, 0].hist(FE_vae_b2, color='tab:olive', edgecolor='black')
axs[2, 0].axvline(b2True, color='black', linewidth=3)
axs[2, 0].set_xlabel('VAE')
axs[2, 0].set_ylabel('Frequency')
# Plot 7
axs[2, 1].hist(FE_gae_b2, color='tab:green', edgecolor='black')
axs[2, 1].axvline(b2True, color='black', linewidth=3)
axs[2, 1].set_xlabel('GAE')
axs[2, 1].set_ylabel('Frequency')
# Plot 8
axs[2, 2].hist(FE_dif_b2, color='tab:brown', edgecolor='black')
axs[2, 2].axvline(b2True, color='black', linewidth=3)
axs[2, 2].set_xlabel('Diffusion')
axs[2, 2].set_ylabel('Frequency')

fig.suptitle('x2 Coefficient - Study Fixed Effects', fontsize=20)
#plt.savefig(filepath+"Figures/Case"+str(case)+"/BiasB2_FEs.png")
plt.savefig(filepath+"/BiasB2_FEs.png")
plt.close()

###

fig, axs = plt.subplots(3, 3, figsize=(12, 16))
plt.subplots_adjust(hspace=0.4, wspace=0.4)
# Plot 1
axs[0, 0].hist(FE_cca_b3, color='tab:cyan', edgecolor='black')
axs[0, 0].axvline(b3True, color='black', linewidth=3)
axs[0, 0].set_xlabel('CCA')
axs[0, 0].set_ylabel('Frequency')
# Plot 2
axs[0, 1].hist(FE_mi_b3, color='tab:blue', edgecolor='black')
axs[0, 1].axvline(b3True, color='black', linewidth=3)
axs[0, 1].set_xlabel('MI')
axs[0, 1].set_ylabel('Frequency')
# Plot 3
axs[0, 2].hist(FE_lh_b3, color='tab:purple', edgecolor='black')
axs[0, 2].axvline(b3True, color='black', linewidth=3)
axs[0, 2].set_xlabel('Likelihood')
axs[0, 2].set_ylabel('Frequency')
# Plot 4
axs[1, 0].hist(FE_rf_b3, color='tab:pink', edgecolor='black')
axs[1, 0].axvline(b3True, color='black', linewidth=3)
axs[1, 0].set_xlabel('Random Forest')
axs[1, 0].set_ylabel('Frequency')
# Plot 5
axs[1, 1].hist(FE_lgb_b3, color='tab:orange', edgecolor='black')
axs[1, 1].axvline(b3True, color='black', linewidth=3)
axs[1, 1].set_xlabel('LGBM')
axs[1, 1].set_ylabel('Frequency')
# Plot 5.5
axs[1, 2].hist(FE_mlp_b3, color='tab:red', edgecolor='black')
axs[1, 2].axvline(b3True, color='black', linewidth=3)
axs[1, 2].set_xlabel('MLP')
axs[1, 2].set_ylabel('Frequency')
# Plot 6
axs[2, 0].hist(FE_vae_b3, color='tab:olive', edgecolor='black')
axs[2, 0].axvline(b3True, color='black', linewidth=3)
axs[2, 0].set_xlabel('VAE')
axs[2, 0].set_ylabel('Frequency')
# Plot 7
axs[2, 1].hist(FE_gae_b3, color='tab:green', edgecolor='black')
axs[2, 1].axvline(b3True, color='black', linewidth=3)
axs[2, 1].set_xlabel('GAE')
axs[2, 1].set_ylabel('Frequency')
# Plot 8
axs[2, 2].hist(FE_dif_b3, color='tab:brown', edgecolor='black')
axs[2, 2].axvline(b3True, color='black', linewidth=3)
axs[2, 2].set_xlabel('Diffusion')
axs[2, 2].set_ylabel('Frequency')

fig.suptitle('x3 Coefficient - Study Fixed Effects', fontsize=20)
#plt.savefig(filepath+"Figures/Case"+str(case)+"/BiasB3_FEs.png")
plt.savefig(filepath+"/BiasB3_FEs.png")
plt.close()

### VARIANCE ###

fig, axs = plt.subplots(3, 3, figsize=(12, 16))
plt.subplots_adjust(hspace=0.4, wspace=0.4)
# Plot 1
axs[0, 0].hist(FE_cca_b1Var, color='tab:cyan', edgecolor='black')
axs[0, 0].set_xlabel('CCA')
axs[0, 0].set_ylabel('Frequency')
# Plot 2
axs[0, 1].hist(FE_mi_b1Var, color='tab:blue', edgecolor='black')
axs[0, 1].set_xlabel('MI')
axs[0, 1].set_ylabel('Frequency')
# Plot 3
axs[0, 2].hist(FE_lh_b1Var, color='tab:purple', edgecolor='black')
axs[0, 2].set_xlabel('Likelihood')
axs[0, 2].set_ylabel('Frequency')
# Plot 4
axs[1, 0].hist(FE_rf_b1Var, color='tab:pink', edgecolor='black')
axs[1, 0].set_xlabel('Random Forest')
axs[1, 0].set_ylabel('Frequency')
# Plot 5
axs[1, 1].hist(FE_lgb_b1Var, color='tab:orange', edgecolor='black')
axs[1, 1].set_xlabel('LGBM')
axs[1, 1].set_ylabel('Frequency')
# Plot 5.5
axs[1, 2].hist(FE_mlp_b1Var, color='tab:red', edgecolor='black')
axs[1, 2].set_xlabel('MLP')
axs[1, 2].set_ylabel('Frequency')
# Plot 6
axs[2, 0].hist(FE_vae_b1Var, color='tab:olive', edgecolor='black')
axs[2, 0].set_xlabel('VAE')
axs[2, 0].set_ylabel('Frequency')
# Plot 7
axs[2, 1].hist(FE_gae_b1Var, color='tab:green', edgecolor='black')
axs[2, 1].set_xlabel('GAE')
axs[2, 1].set_ylabel('Frequency')
# Plot 8
axs[2, 2].hist(FE_dif_b1Var, color='tab:brown', edgecolor='black')
axs[2, 2].set_xlabel('Diffusion')
axs[2, 2].set_ylabel('Frequency')

fig.suptitle('x1 Variance - Study Fixed Effects', fontsize=20)
#plt.savefig(filepath+"Figures/Case"+str(case)+"/VarB1_FEs.png")
plt.savefig(filepath+"/VarB1_FEs.png")
plt.close()

###

fig, axs = plt.subplots(3, 3, figsize=(12, 16))
plt.subplots_adjust(hspace=0.4, wspace=0.4)
# Plot 1
axs[0, 0].hist(FE_cca_b2Var, color='tab:cyan', edgecolor='black')
axs[0, 0].set_xlabel('CCA')
axs[0, 0].set_ylabel('Frequency')
# Plot 2
axs[0, 1].hist(FE_mi_b2Var, color='tab:blue', edgecolor='black')
axs[0, 1].set_xlabel('MI')
axs[0, 1].set_ylabel('Frequency')
# Plot 3
axs[0, 2].hist(FE_lh_b2Var, color='tab:purple', edgecolor='black')
axs[0, 2].set_xlabel('Likelihood')
axs[0, 2].set_ylabel('Frequency')
# Plot 4
axs[1, 0].hist(FE_rf_b2Var, color='tab:pink', edgecolor='black')
axs[1, 0].set_xlabel('Random Forest')
axs[1, 0].set_ylabel('Frequency')
# Plot 5
axs[1, 1].hist(FE_lgb_b2Var, color='tab:orange', edgecolor='black')
axs[1, 1].set_xlabel('LGBM')
axs[1, 1].set_ylabel('Frequency')
# Plot 5.5
axs[1, 2].hist(FE_mlp_b2Var, color='tab:red', edgecolor='black')
axs[1, 2].set_xlabel('MLP')
axs[1, 2].set_ylabel('Frequency')
# Plot 6
axs[2, 0].hist(FE_vae_b2Var, color='tab:olive', edgecolor='black')
axs[2, 0].set_xlabel('VAE')
axs[2, 0].set_ylabel('Frequency')
# Plot 7
axs[2, 1].hist(FE_gae_b2Var, color='tab:green', edgecolor='black')
axs[2, 1].set_xlabel('GAE')
axs[2, 1].set_ylabel('Frequency')
# Plot 8
axs[2, 2].hist(FE_dif_b2Var, color='tab:brown', edgecolor='black')
axs[2, 2].set_xlabel('Diffusion')
axs[2, 2].set_ylabel('Frequency')

fig.suptitle('x2 Variance - Study Fixed Effects', fontsize=20)
#plt.savefig(filepath+"Figures/Case"+str(case)+"/VarB2_FEs.png")
plt.savefig(filepath+"/VarB2_FEs.png")
plt.close()

###

fig, axs = plt.subplots(3, 3, figsize=(12, 16))
plt.subplots_adjust(hspace=0.4, wspace=0.4)
# Plot 1
axs[0, 0].hist(FE_cca_b3Var, color='tab:cyan', edgecolor='black')
axs[0, 0].set_xlabel('CCA')
axs[0, 0].set_ylabel('Frequency')
# Plot 2
axs[0, 1].hist(FE_mi_b3Var, color='tab:blue', edgecolor='black')
axs[0, 1].set_xlabel('MI')
axs[0, 1].set_ylabel('Frequency')
# Plot 3
axs[0, 2].hist(FE_lh_b3Var, color='tab:purple', edgecolor='black')
axs[0, 2].set_xlabel('Likelihood')
axs[0, 2].set_ylabel('Frequency')
# Plot 4
axs[1, 0].hist(FE_rf_b3Var, color='tab:pink', edgecolor='black')
axs[1, 0].set_xlabel('Random Forest')
axs[1, 0].set_ylabel('Frequency')
# Plot 5
axs[1, 1].hist(FE_lgb_b3Var, color='tab:orange', edgecolor='black')
axs[1, 1].set_xlabel('LGBM')
axs[1, 1].set_ylabel('Frequency')
# Plot 5.5
axs[1, 2].hist(FE_mlp_b3Var, color='tab:red', edgecolor='black')
axs[1, 2].set_xlabel('MLP')
axs[1, 2].set_ylabel('Frequency')
# Plot 6
axs[2, 0].hist(FE_vae_b3Var, color='tab:olive', edgecolor='black')
axs[2, 0].set_xlabel('VAE')
axs[2, 0].set_ylabel('Frequency')
# Plot 7
axs[2, 1].hist(FE_gae_b3Var, color='tab:green', edgecolor='black')
axs[2, 1].set_xlabel('GAE')
axs[2, 1].set_ylabel('Frequency')
# Plot 8
axs[2, 2].hist(FE_dif_b3Var, color='tab:brown', edgecolor='black')
axs[2, 2].set_xlabel('Diffusion')
axs[2, 2].set_ylabel('Frequency')

fig.suptitle('x3 Variance - Study Fixed Effects', fontsize=20)
#plt.savefig(filepath+"Figures/Case"+str(case)+"/VarB3_FEs.png")
plt.savefig(filepath+"/VarB3_FEs.png")
plt.close()

### MSE ###

fig, axs = plt.subplots(3, 3, figsize=(12, 16))
plt.subplots_adjust(hspace=0.4, wspace=0.4)
# Plot 1
axs[0, 0].hist(FE_cca_mse, color='tab:cyan', edgecolor='black')
axs[0, 0].set_xlabel('CCA')
axs[0, 0].set_ylabel('Frequency')
# Plot 2
axs[0, 1].hist(FE_mi_mse, color='tab:blue', edgecolor='black')
axs[0, 1].set_xlabel('MI')
axs[0, 1].set_ylabel('Frequency')
# Plot 3
axs[0, 2].hist(FE_lh_mse, color='tab:purple', edgecolor='black')
axs[0, 2].set_xlabel('Likelihood')
axs[0, 2].set_ylabel('Frequency')
# Plot 4
axs[1, 0].hist(FE_rf_mse, color='tab:pink', edgecolor='black')
axs[1, 0].set_xlabel('Random Forest')
axs[1, 0].set_ylabel('Frequency')
# Plot 5
axs[1, 1].hist(FE_lgb_mse, color='tab:orange', edgecolor='black')
axs[1, 1].set_xlabel('LGBM')
axs[1, 1].set_ylabel('Frequency')
# Plot 5.5
axs[1, 2].hist(FE_mlp_mse, color='tab:red', edgecolor='black')
axs[1, 2].set_xlabel('MLP')
axs[1, 2].set_ylabel('Frequency')
# Plot 6
axs[2, 0].hist(FE_vae_mse, color='tab:olive', edgecolor='black')
axs[2, 0].set_xlabel('VAE')
axs[2, 0].set_ylabel('Frequency')
# Plot 7
axs[2, 1].hist(FE_gae_mse, color='tab:green', edgecolor='black')
axs[2, 1].set_xlabel('GAE')
axs[2, 1].set_ylabel('Frequency')
# Plot 8
axs[2, 2].hist(FE_dif_mse, color='tab:brown', edgecolor='black')
axs[2, 2].set_xlabel('Diffusion')
axs[2, 2].set_ylabel('Frequency')

fig.suptitle('y MSE - Study Fixed Effects', fontsize=20)
#plt.savefig(filepath+"Figures/Case"+str(case)+"/mse_FEs.png")
plt.savefig(filepath+"/mse_FEs.png")
plt.close()

###

fig, axs = plt.subplots(3, 3, figsize=(12, 16))
plt.subplots_adjust(hspace=0.4, wspace=0.4)
# Plot 1
axs[0, 0].hist(FE_cca_mse_x1, color='tab:cyan', edgecolor='black')
axs[0, 0].set_xlabel('CCA')
axs[0, 0].set_ylabel('Frequency')
# Plot 2
axs[0, 1].hist(FE_mi_mse_x1, color='tab:blue', edgecolor='black')
axs[0, 1].set_xlabel('MI')
axs[0, 1].set_ylabel('Frequency')
# Plot 3
axs[0, 2].hist(FE_lh_mse_x1, color='tab:purple', edgecolor='black')
axs[0, 2].set_xlabel('Likelihood')
axs[0, 2].set_ylabel('Frequency')
# Plot 4
axs[1, 0].hist(FE_rf_mse_x1, color='tab:pink', edgecolor='black')
axs[1, 0].set_xlabel('Random Forest')
axs[1, 0].set_ylabel('Frequency')
# Plot 5
axs[1, 1].hist(FE_lgb_mse_x1, color='tab:orange', edgecolor='black')
axs[1, 1].set_xlabel('LGBM')
axs[1, 1].set_ylabel('Frequency')
# Plot 5.5
axs[1, 2].hist(FE_mlp_mse_x1, color='tab:red', edgecolor='black')
axs[1, 2].set_xlabel('MLP')
axs[1, 2].set_ylabel('Frequency')
# Plot 6
axs[2, 0].hist(FE_vae_mse_x1, color='tab:olive', edgecolor='black')
axs[2, 0].set_xlabel('VAE')
axs[2, 0].set_ylabel('Frequency')
# Plot 7
axs[2, 1].hist(FE_gae_mse_x1, color='tab:green', edgecolor='black')
axs[2, 1].set_xlabel('GAE')
axs[2, 1].set_ylabel('Frequency')
# Plot 8
axs[2, 2].hist(FE_dif_mse_x1, color='tab:brown', edgecolor='black')
axs[2, 2].set_xlabel('Diffusion')
axs[2, 2].set_ylabel('Frequency')

fig.suptitle('x1 MSE - Study Fixed Effects', fontsize=20)
#plt.savefig(filepath+"Figures/Case"+str(case)+"/mseX1_FEs.png")
plt.savefig(filepath+"/mseX1_FEs.png")
plt.close()

###

fig, axs = plt.subplots(3, 3, figsize=(12, 16))
plt.subplots_adjust(hspace=0.4, wspace=0.4)
# Plot 1
axs[0, 0].hist(FE_cca_mse_x2, color='tab:cyan', edgecolor='black')
axs[0, 0].set_xlabel('CCA')
axs[0, 0].set_ylabel('Frequency')
# Plot 2
axs[0, 1].hist(FE_mi_mse_x2, color='tab:blue', edgecolor='black')
axs[0, 1].set_xlabel('MI')
axs[0, 1].set_ylabel('Frequency')
# Plot 3
axs[0, 2].hist(FE_lh_mse_x2, color='tab:purple', edgecolor='black')
axs[0, 2].set_xlabel('Likelihood')
axs[0, 2].set_ylabel('Frequency')
# Plot 4
axs[1, 0].hist(FE_rf_mse_x2, color='tab:pink', edgecolor='black')
axs[1, 0].set_xlabel('Random Forest')
axs[1, 0].set_ylabel('Frequency')
# Plot 5
axs[1, 1].hist(FE_lgb_mse_x2, color='tab:orange', edgecolor='black')
axs[1, 1].set_xlabel('LGBM')
axs[1, 1].set_ylabel('Frequency')
# Plot 5.5
axs[1, 2].hist(FE_mlp_mse_x2, color='tab:red', edgecolor='black')
axs[1, 2].set_xlabel('MLP')
axs[1, 2].set_ylabel('Frequency')
# Plot 6
axs[2, 0].hist(FE_vae_mse_x2, color='tab:olive', edgecolor='black')
axs[2, 0].set_xlabel('VAE')
axs[2, 0].set_ylabel('Frequency')
# Plot 7
axs[2, 1].hist(FE_gae_mse_x2, color='tab:green', edgecolor='black')
axs[2, 1].set_xlabel('GAE')
axs[2, 1].set_ylabel('Frequency')
# Plot 8
axs[2, 2].hist(FE_dif_mse_x2, color='tab:brown', edgecolor='black')
axs[2, 2].set_xlabel('Diffusion')
axs[2, 2].set_ylabel('Frequency')

fig.suptitle('x2 MSE - Study Fixed Effects', fontsize=20)
#plt.savefig(filepath+"Figures/Case"+str(case)+"/mseX2_FEs.png")
plt.savefig(filepath+"/mseX2_FEs.png")
plt.close()

###

fig, axs = plt.subplots(3, 3, figsize=(12, 16))
plt.subplots_adjust(hspace=0.4, wspace=0.4)
# Plot 1
axs[0, 0].hist(FE_cca_mse_x3, color='tab:cyan', edgecolor='black')
axs[0, 0].set_xlabel('CCA')
axs[0, 0].set_ylabel('Frequency')
# Plot 2
axs[0, 1].hist(FE_mi_mse_x3, color='tab:blue', edgecolor='black')
axs[0, 1].set_xlabel('MI')
axs[0, 1].set_ylabel('Frequency')
# Plot 3
axs[0, 2].hist(FE_lh_mse_x3, color='tab:purple', edgecolor='black')
axs[0, 2].set_xlabel('Likelihood')
axs[0, 2].set_ylabel('Frequency')
# Plot 4
axs[1, 0].hist(FE_rf_mse_x3, color='tab:pink', edgecolor='black')
axs[1, 0].set_xlabel('Random Forest')
axs[1, 0].set_ylabel('Frequency')
# Plot 5
axs[1, 1].hist(FE_lgb_mse_x3, color='tab:orange', edgecolor='black')
axs[1, 1].set_xlabel('LGBM')
axs[1, 1].set_ylabel('Frequency')
# Plot 5.5
axs[1, 2].hist(FE_mlp_mse_x3, color='tab:red', edgecolor='black')
axs[1, 2].set_xlabel('MLP')
axs[1, 2].set_ylabel('Frequency')
# Plot 6
axs[2, 0].hist(FE_vae_mse_x3, color='tab:olive', edgecolor='black')
axs[2, 0].set_xlabel('VAE')
axs[2, 0].set_ylabel('Frequency')
# Plot 7
axs[2, 1].hist(FE_gae_mse_x3, color='tab:green', edgecolor='black')
axs[2, 1].set_xlabel('GAE')
axs[2, 1].set_ylabel('Frequency')
# Plot 8
axs[2, 2].hist(FE_dif_mse_x3, color='tab:brown', edgecolor='black')
axs[2, 2].set_xlabel('Diffusion')
axs[2, 2].set_ylabel('Frequency')

fig.suptitle('x3 MSE - Study Fixed Effects', fontsize=20)
#plt.savefig(filepath+"Figures/Case"+str(case)+"/mseX3_FEs.png")
plt.savefig(filepath+"/mseX3_FEs.png")
plt.close()



###### RE ######
### BIAS ###

fig, axs = plt.subplots(3, 3, figsize=(12, 16))
plt.subplots_adjust(hspace=0.4, wspace=0.4)
# Plot 1
axs[0, 0].hist(RE_cca_b0, color='tab:cyan', edgecolor='black')
#axs[0, 0].axvline(b0True, color='black', linewidth=3)
axs[0, 0].set_xlabel('CCA')
axs[0, 0].set_ylabel('Frequency')
# Plot 2
axs[0, 1].hist(RE_mi_b0, color='tab:blue', edgecolor='black')
#axs[0, 1].axvline(b0True, color='black', linewidth=3)
axs[0, 1].set_xlabel('MI')
axs[0, 1].set_ylabel('Frequency')
# Plot 3
axs[0, 2].hist(RE_lh_b0, color='tab:purple', edgecolor='black')
#axs[0, 2].axvline(b0True, color='black', linewidth=3)
axs[0, 2].set_xlabel('Likelihood')
axs[0, 2].set_ylabel('Frequency')
# Plot 4
axs[1, 0].hist(RE_rf_b0, color='tab:pink', edgecolor='black')
#axs[1, 0].axvline(b0True, color='black', linewidth=3)
axs[1, 0].set_xlabel('Random Forest')
axs[1, 0].set_ylabel('Frequency')
# Plot 5
axs[1, 1].hist(RE_lgb_b0, color='tab:orange', edgecolor='black')
#axs[1, 1].axvline(b0True, color='black', linewidth=3)
axs[1, 1].set_xlabel('LGBM')
axs[1, 1].set_ylabel('Frequency')
# Plot 5.5
axs[1, 2].hist(RE_mlp_b0, color='tab:red', edgecolor='black')
#axs[1, 2].axvline(b0True, color='black', linewidth=3)
axs[1, 2].set_xlabel('MLP')
axs[1, 2].set_ylabel('Frequency')
# Plot 6
axs[2, 0].hist(RE_vae_b0, color='tab:olive', edgecolor='black')
#axs[2, 0].axvline(b0True, color='black', linewidth=3)
axs[2, 0].set_xlabel('VAE')
axs[2, 0].set_ylabel('Frequency')
# Plot 7
axs[2, 1].hist(RE_gae_b0, color='tab:green', edgecolor='black')
#axs[2, 1].axvline(b0True, color='black', linewidth=3)
axs[2, 1].set_xlabel('GAE')
axs[2, 1].set_ylabel('Frequency')
# Plot 8
axs[2, 2].hist(RE_dif_b0, color='tab:brown', edgecolor='black')
#axs[2, 2].axvline(b0True, color='black', linewidth=3)
axs[2, 2].set_xlabel('Diffusion')
axs[2, 2].set_ylabel('Frequency')

fig.suptitle('Intercept Coefficient - Study Random Effects', fontsize=20)
#plt.savefig(filepath+"Figures/Case"+str(case)+"/BiasB0_RE.png")
plt.savefig(filepath+"/BiasB0_RE.png")
plt.close()

###

fig, axs = plt.subplots(3, 3, figsize=(12, 16))
plt.subplots_adjust(hspace=0.4, wspace=0.4)
# Plot 1
axs[0, 0].hist(RE_cca_b1, color='tab:cyan', edgecolor='black')
axs[0, 0].axvline(b1True, color='black', linewidth=3)
axs[0, 0].set_xlabel('CCA')
axs[0, 0].set_ylabel('Frequency')
# Plot 2
axs[0, 1].hist(RE_mi_b1, color='tab:blue', edgecolor='black')
axs[0, 1].axvline(b1True, color='black', linewidth=3)
axs[0, 1].set_xlabel('MI')
axs[0, 1].set_ylabel('Frequency')
# Plot 3
axs[0, 2].hist(RE_lh_b1, color='tab:purple', edgecolor='black')
axs[0, 2].axvline(b1True, color='black', linewidth=3)
axs[0, 2].set_xlabel('Likelihood')
axs[0, 2].set_ylabel('Frequency')
# Plot 4
axs[1, 0].hist(RE_rf_b1, color='tab:pink', edgecolor='black')
axs[1, 0].axvline(b1True, color='black', linewidth=3)
axs[1, 0].set_xlabel('Random Forest')
axs[1, 0].set_ylabel('Frequency')
# Plot 5
axs[1, 1].hist(RE_lgb_b1, color='tab:orange', edgecolor='black')
axs[1, 1].axvline(b1True, color='black', linewidth=3)
axs[1, 1].set_xlabel('LGBM')
axs[1, 1].set_ylabel('Frequency')
# Plot 5.5
axs[1, 2].hist(RE_mlp_b1, color='tab:red', edgecolor='black')
axs[1, 2].axvline(b1True, color='black', linewidth=3)
axs[1, 2].set_xlabel('MLP')
axs[1, 2].set_ylabel('Frequency')
# Plot 6
axs[2, 0].hist(RE_vae_b1, color='tab:olive', edgecolor='black')
axs[2, 0].axvline(b1True, color='black', linewidth=3)
axs[2, 0].set_xlabel('VAE')
axs[2, 0].set_ylabel('Frequency')
# Plot 7
axs[2, 1].hist(RE_gae_b1, color='tab:green', edgecolor='black')
axs[2, 1].axvline(b1True, color='black', linewidth=3)
axs[2, 1].set_xlabel('GAE')
axs[2, 1].set_ylabel('Frequency')
# Plot 8
axs[2, 2].hist(RE_dif_b1, color='tab:brown', edgecolor='black')
axs[2, 2].axvline(b1True, color='black', linewidth=3)
axs[2, 2].set_xlabel('Diffusion')
axs[2, 2].set_ylabel('Frequency')

fig.suptitle('x1 Coefficient - Study Random Effects', fontsize=20)
#plt.savefig(filepath+"Figures/Case"+str(case)+"/BiasB1_RE.png")
plt.savefig(filepath+"/BiasB1_RE.png")
plt.close()

###

fig, axs = plt.subplots(3, 3, figsize=(12, 16))
plt.subplots_adjust(hspace=0.4, wspace=0.4)
# Plot 1
axs[0, 0].hist(RE_cca_b2, color='tab:cyan', edgecolor='black')
axs[0, 0].axvline(b2True, color='black', linewidth=3)
axs[0, 0].set_xlabel('CCA')
axs[0, 0].set_ylabel('Frequency')
# Plot 2
axs[0, 1].hist(RE_mi_b2, color='tab:blue', edgecolor='black')
axs[0, 1].axvline(b2True, color='black', linewidth=3)
axs[0, 1].set_xlabel('MI')
axs[0, 1].set_ylabel('Frequency')
# Plot 3
axs[0, 2].hist(RE_lh_b2, color='tab:purple', edgecolor='black')
axs[0, 2].axvline(b2True, color='black', linewidth=3)
axs[0, 2].set_xlabel('Likelihood')
axs[0, 2].set_ylabel('Frequency')
# Plot 4
axs[1, 0].hist(RE_rf_b2, color='tab:pink', edgecolor='black')
axs[1, 0].axvline(b2True, color='black', linewidth=3)
axs[1, 0].set_xlabel('Random Forest')
axs[1, 0].set_ylabel('Frequency')
# Plot 5
axs[1, 1].hist(RE_lgb_b2, color='tab:orange', edgecolor='black')
axs[1, 1].axvline(b2True, color='black', linewidth=3)
axs[1, 1].set_xlabel('LGBM')
axs[1, 1].set_ylabel('Frequency')
# Plot 5.5
axs[1, 2].hist(RE_mlp_b2, color='tab:red', edgecolor='black')
axs[1, 2].axvline(b2True, color='black', linewidth=3)
axs[1, 2].set_xlabel('MLP')
axs[1, 2].set_ylabel('Frequency')
# Plot 6
axs[2, 0].hist(RE_vae_b2, color='tab:olive', edgecolor='black')
axs[2, 0].axvline(b2True, color='black', linewidth=3)
axs[2, 0].set_xlabel('VAE')
axs[2, 0].set_ylabel('Frequency')
# Plot 7
axs[2, 1].hist(RE_gae_b2, color='tab:green', edgecolor='black')
axs[2, 1].axvline(b2True, color='black', linewidth=3)
axs[2, 1].set_xlabel('GAE')
axs[2, 1].set_ylabel('Frequency')
# Plot 8
axs[2, 2].hist(RE_dif_b2, color='tab:brown', edgecolor='black')
axs[2, 2].axvline(b2True, color='black', linewidth=3)
axs[2, 2].set_xlabel('Diffusion')
axs[2, 2].set_ylabel('Frequency')

fig.suptitle('x2 Coefficient - Study Random Effects', fontsize=20)
#plt.savefig(filepath+"Figures/Case"+str(case)+"/BiasB2_RE.png")
plt.savefig(filepath+"/BiasB2_RE.png")
plt.close()

###

fig, axs = plt.subplots(3, 3, figsize=(12, 16))
plt.subplots_adjust(hspace=0.4, wspace=0.4)
# Plot 1
axs[0, 0].hist(RE_cca_b3, color='tab:cyan', edgecolor='black')
axs[0, 0].axvline(b3True, color='black', linewidth=3)
axs[0, 0].set_xlabel('CCA')
axs[0, 0].set_ylabel('Frequency')
# Plot 2
axs[0, 1].hist(RE_mi_b3, color='tab:blue', edgecolor='black')
axs[0, 1].axvline(b3True, color='black', linewidth=3)
axs[0, 1].set_xlabel('MI')
axs[0, 1].set_ylabel('Frequency')
# Plot 3
axs[0, 2].hist(RE_lh_b3, color='tab:purple', edgecolor='black')
axs[0, 2].axvline(b3True, color='black', linewidth=3)
axs[0, 2].set_xlabel('Likelihood')
axs[0, 2].set_ylabel('Frequency')
# Plot 4
axs[1, 0].hist(RE_rf_b3, color='tab:pink', edgecolor='black')
axs[1, 0].axvline(b3True, color='black', linewidth=3)
axs[1, 0].set_xlabel('Random Forest')
axs[1, 0].set_ylabel('Frequency')
# Plot 5
axs[1, 1].hist(RE_lgb_b3, color='tab:orange', edgecolor='black')
axs[1, 1].axvline(b3True, color='black', linewidth=3)
axs[1, 1].set_xlabel('LGBM')
axs[1, 1].set_ylabel('Frequency')
# Plot 5.5
axs[1, 2].hist(RE_mlp_b3, color='tab:red', edgecolor='black')
axs[1, 2].axvline(b3True, color='black', linewidth=3)
axs[1, 2].set_xlabel('MLP')
axs[1, 2].set_ylabel('Frequency')
# Plot 6
axs[2, 0].hist(RE_vae_b3, color='tab:olive', edgecolor='black')
axs[2, 0].axvline(b3True, color='black', linewidth=3)
axs[2, 0].set_xlabel('VAE')
axs[2, 0].set_ylabel('Frequency')
# Plot 7
axs[2, 1].hist(RE_gae_b3, color='tab:green', edgecolor='black')
axs[2, 1].axvline(b3True, color='black', linewidth=3)
axs[2, 1].set_xlabel('GAE')
axs[2, 1].set_ylabel('Frequency')
# Plot 8
axs[2, 2].hist(RE_dif_b3, color='tab:brown', edgecolor='black')
axs[2, 2].axvline(b3True, color='black', linewidth=3)
axs[2, 2].set_xlabel('Diffusion')
axs[2, 2].set_ylabel('Frequency')

fig.suptitle('x3 Coefficient - Study Random Effects', fontsize=20)
#plt.savefig(filepath+"Figures/Case"+str(case)+"/BiasB3_RE.png")
plt.savefig(filepath+"/BiasB3_RE.png")
plt.close()

### VARIANCE ###

fig, axs = plt.subplots(3, 3, figsize=(12, 16))
plt.subplots_adjust(hspace=0.4, wspace=0.4)
# Plot 1
axs[0, 0].hist(RE_cca_b1Var, color='tab:cyan', edgecolor='black')
axs[0, 0].set_xlabel('CCA')
axs[0, 0].set_ylabel('Frequency')
# Plot 2
axs[0, 1].hist(RE_mi_b1Var, color='tab:blue', edgecolor='black')
axs[0, 1].set_xlabel('MI')
axs[0, 1].set_ylabel('Frequency')
# Plot 3
axs[0, 2].hist(RE_lh_b1Var, color='tab:purple', edgecolor='black')
axs[0, 2].set_xlabel('Likelihood')
axs[0, 2].set_ylabel('Frequency')
# Plot 4
axs[1, 0].hist(RE_rf_b1Var, color='tab:pink', edgecolor='black')
axs[1, 0].set_xlabel('Random Forest')
axs[1, 0].set_ylabel('Frequency')
# Plot 5
axs[1, 1].hist(RE_lgb_b1Var, color='tab:orange', edgecolor='black')
axs[1, 1].set_xlabel('LGBM')
axs[1, 1].set_ylabel('Frequency')
# Plot 5.5
axs[1, 2].hist(RE_mlp_b1Var, color='tab:red', edgecolor='black')
axs[1, 2].set_xlabel('MLP')
axs[1, 2].set_ylabel('Frequency')
# Plot 6
axs[2, 0].hist(RE_vae_b1Var, color='tab:olive', edgecolor='black')
axs[2, 0].set_xlabel('VAE')
axs[2, 0].set_ylabel('Frequency')
# Plot 7
axs[2, 1].hist(RE_gae_b1Var, color='tab:green', edgecolor='black')
axs[2, 1].set_xlabel('GAE')
axs[2, 1].set_ylabel('Frequency')
# Plot 8
axs[2, 2].hist(RE_dif_b1Var, color='tab:brown', edgecolor='black')
axs[2, 2].set_xlabel('Diffusion')
axs[2, 2].set_ylabel('Frequency')

fig.suptitle('x1 Variance - Study Random Effects', fontsize=20)
#plt.savefig(filepath+"Figures/Case"+str(case)+"/VarB1_RE.png")
plt.savefig(filepath+"/VarB1_RE.png")
plt.close()

###

fig, axs = plt.subplots(3, 3, figsize=(12, 16))
plt.subplots_adjust(hspace=0.4, wspace=0.4)
# Plot 1
axs[0, 0].hist(RE_cca_b2Var, color='tab:cyan', edgecolor='black')
axs[0, 0].set_xlabel('CCA')
axs[0, 0].set_ylabel('Frequency')
# Plot 2
axs[0, 1].hist(RE_mi_b2Var, color='tab:blue', edgecolor='black')
axs[0, 1].set_xlabel('MI')
axs[0, 1].set_ylabel('Frequency')
# Plot 3
axs[0, 2].hist(RE_lh_b2Var, color='tab:purple', edgecolor='black')
axs[0, 2].set_xlabel('Likelihood')
axs[0, 2].set_ylabel('Frequency')
# Plot 4
axs[1, 0].hist(RE_rf_b2Var, color='tab:pink', edgecolor='black')
axs[1, 0].set_xlabel('Random Forest')
axs[1, 0].set_ylabel('Frequency')
# Plot 5
axs[1, 1].hist(RE_lgb_b2Var, color='tab:orange', edgecolor='black')
axs[1, 1].set_xlabel('LGBM')
axs[1, 1].set_ylabel('Frequency')
# Plot 5.5
axs[1, 2].hist(RE_mlp_b2Var, color='tab:red', edgecolor='black')
axs[1, 2].set_xlabel('MLP')
axs[1, 2].set_ylabel('Frequency')
# Plot 6
axs[2, 0].hist(RE_vae_b2Var, color='tab:olive', edgecolor='black')
axs[2, 0].set_xlabel('VAE')
axs[2, 0].set_ylabel('Frequency')
# Plot 7
axs[2, 1].hist(RE_gae_b2Var, color='tab:green', edgecolor='black')
axs[2, 1].set_xlabel('GAE')
axs[2, 1].set_ylabel('Frequency')
# Plot 8
axs[2, 2].hist(RE_dif_b2Var, color='tab:brown', edgecolor='black')
axs[2, 2].set_xlabel('Diffusion')
axs[2, 2].set_ylabel('Frequency')

fig.suptitle('x2 Variance - Study Random Effects', fontsize=20)
#plt.savefig(filepath+"Figures/Case"+str(case)+"/VarB2_RE.png")
plt.savefig(filepath+"/VarB2_RE.png")
plt.close()

###

fig, axs = plt.subplots(3, 3, figsize=(12, 16))
plt.subplots_adjust(hspace=0.4, wspace=0.4)
# Plot 1
axs[0, 0].hist(RE_cca_b3Var, color='tab:cyan', edgecolor='black')
axs[0, 0].set_xlabel('CCA')
axs[0, 0].set_ylabel('Frequency')
# Plot 2
axs[0, 1].hist(RE_mi_b3Var, color='tab:blue', edgecolor='black')
axs[0, 1].set_xlabel('MI')
axs[0, 1].set_ylabel('Frequency')
# Plot 3
axs[0, 2].hist(RE_lh_b3Var, color='tab:purple', edgecolor='black')
axs[0, 2].set_xlabel('Likelihood')
axs[0, 2].set_ylabel('Frequency')
# Plot 4
axs[1, 0].hist(RE_rf_b3Var, color='tab:pink', edgecolor='black')
axs[1, 0].set_xlabel('Random Forest')
axs[1, 0].set_ylabel('Frequency')
# Plot 5
axs[1, 1].hist(RE_lgb_b3Var, color='tab:orange', edgecolor='black')
axs[1, 1].set_xlabel('LGBM')
axs[1, 1].set_ylabel('Frequency')
# Plot 5.5
axs[1, 2].hist(RE_mlp_b3Var, color='tab:red', edgecolor='black')
axs[1, 2].set_xlabel('MLP')
axs[1, 2].set_ylabel('Frequency')
# Plot 6
axs[2, 0].hist(RE_vae_b3Var, color='tab:olive', edgecolor='black')
axs[2, 0].set_xlabel('VAE')
axs[2, 0].set_ylabel('Frequency')
# Plot 7
axs[2, 1].hist(RE_gae_b3Var, color='tab:green', edgecolor='black')
axs[2, 1].set_xlabel('GAE')
axs[2, 1].set_ylabel('Frequency')
# Plot 8
axs[2, 2].hist(RE_dif_b3Var, color='tab:brown', edgecolor='black')
axs[2, 2].set_xlabel('Diffusion')
axs[2, 2].set_ylabel('Frequency')

fig.suptitle('x3 Variance - Study Random Effects', fontsize=20)
#plt.savefig(filepath+"Figures/Case"+str(case)+"/VarB3_RE.png")
plt.savefig(filepath+"/VarB3_RE.png")
plt.close()

### MSE ###

fig, axs = plt.subplots(3, 3, figsize=(12, 16))
plt.subplots_adjust(hspace=0.4, wspace=0.4)
# Plot 1
axs[0, 0].hist(RE_cca_mse, color='tab:cyan', edgecolor='black')
axs[0, 0].set_xlabel('CCA')
axs[0, 0].set_ylabel('Frequency')
# Plot 2
axs[0, 1].hist(RE_mi_mse, color='tab:blue', edgecolor='black')
axs[0, 1].set_xlabel('MI')
axs[0, 1].set_ylabel('Frequency')
# Plot 3
axs[0, 2].hist(RE_lh_mse, color='tab:purple', edgecolor='black')
axs[0, 2].set_xlabel('Likelihood')
axs[0, 2].set_ylabel('Frequency')
# Plot 4
axs[1, 0].hist(RE_rf_mse, color='tab:pink', edgecolor='black')
axs[1, 0].set_xlabel('Random Forest')
axs[1, 0].set_ylabel('Frequency')
# Plot 5
axs[1, 1].hist(RE_lgb_mse, color='tab:orange', edgecolor='black')
axs[1, 1].set_xlabel('LGBM')
axs[1, 1].set_ylabel('Frequency')
# Plot 5.5
axs[1, 2].hist(RE_mlp_mse, color='tab:red', edgecolor='black')
axs[1, 2].set_xlabel('MLP')
axs[1, 2].set_ylabel('Frequency')
# Plot 6
axs[2, 0].hist(RE_vae_mse, color='tab:olive', edgecolor='black')
axs[2, 0].set_xlabel('VAE')
axs[2, 0].set_ylabel('Frequency')
# Plot 7
axs[2, 1].hist(RE_gae_mse, color='tab:green', edgecolor='black')
axs[2, 1].set_xlabel('GAE')
axs[2, 1].set_ylabel('Frequency')
# Plot 8
axs[2, 2].hist(RE_dif_mse, color='tab:brown', edgecolor='black')
axs[2, 2].set_xlabel('Diffusion')
axs[2, 2].set_ylabel('Frequency')

fig.suptitle('y MSE - Study Random Effects', fontsize=20)
#plt.savefig(filepath+"Figures/Case"+str(case)+"/mse_RE.png")
plt.savefig(filepath+"/mse_RE.png")
plt.close()

###

fig, axs = plt.subplots(3, 3, figsize=(12, 16))
plt.subplots_adjust(hspace=0.4, wspace=0.4)
# Plot 1
axs[0, 0].hist(RE_cca_mse_x1, color='tab:cyan', edgecolor='black')
axs[0, 0].set_xlabel('CCA')
axs[0, 0].set_ylabel('Frequency')
# Plot 2
axs[0, 1].hist(RE_mi_mse_x1, color='tab:blue', edgecolor='black')
axs[0, 1].set_xlabel('MI')
axs[0, 1].set_ylabel('Frequency')
# Plot 3
axs[0, 2].hist(RE_lh_mse_x1, color='tab:purple', edgecolor='black')
axs[0, 2].set_xlabel('Likelihood')
axs[0, 2].set_ylabel('Frequency')
# Plot 4
axs[1, 0].hist(RE_rf_mse_x1, color='tab:pink', edgecolor='black')
axs[1, 0].set_xlabel('Random Forest')
axs[1, 0].set_ylabel('Frequency')
# Plot 5
axs[1, 1].hist(RE_lgb_mse_x1, color='tab:orange', edgecolor='black')
axs[1, 1].set_xlabel('LGBM')
axs[1, 1].set_ylabel('Frequency')
# Plot 5.5
axs[1, 2].hist(RE_mlp_mse_x1, color='tab:red', edgecolor='black')
axs[1, 2].set_xlabel('MLP')
axs[1, 2].set_ylabel('Frequency')
# Plot 6
axs[2, 0].hist(RE_vae_mse_x1, color='tab:olive', edgecolor='black')
axs[2, 0].set_xlabel('VAE')
axs[2, 0].set_ylabel('Frequency')
# Plot 7
axs[2, 1].hist(RE_gae_mse_x1, color='tab:green', edgecolor='black')
axs[2, 1].set_xlabel('GAE')
axs[2, 1].set_ylabel('Frequency')
# Plot 8
axs[2, 2].hist(RE_dif_mse_x1, color='tab:brown', edgecolor='black')
axs[2, 2].set_xlabel('Diffusion')
axs[2, 2].set_ylabel('Frequency')

fig.suptitle('x1 MSE - Study Random Effects', fontsize=20)
#plt.savefig(filepath+"Figures/Case"+str(case)+"/mseX1_RE.png")
plt.savefig(filepath+"/mseX1_RE.png")
plt.close()

###

fig, axs = plt.subplots(3, 3, figsize=(12, 16))
plt.subplots_adjust(hspace=0.4, wspace=0.4)
# Plot 1
axs[0, 0].hist(RE_cca_mse_x2, color='tab:cyan', edgecolor='black')
axs[0, 0].set_xlabel('CCA')
axs[0, 0].set_ylabel('Frequency')
# Plot 2
axs[0, 1].hist(RE_mi_mse_x2, color='tab:blue', edgecolor='black')
axs[0, 1].set_xlabel('MI')
axs[0, 1].set_ylabel('Frequency')
# Plot 3
axs[0, 2].hist(RE_lh_mse_x2, color='tab:purple', edgecolor='black')
axs[0, 2].set_xlabel('Likelihood')
axs[0, 2].set_ylabel('Frequency')
# Plot 4
axs[1, 0].hist(RE_rf_mse_x2, color='tab:pink', edgecolor='black')
axs[1, 0].set_xlabel('Random Forest')
axs[1, 0].set_ylabel('Frequency')
# Plot 5
axs[1, 1].hist(RE_lgb_mse_x2, color='tab:orange', edgecolor='black')
axs[1, 1].set_xlabel('LGBM')
axs[1, 1].set_ylabel('Frequency')
# Plot 5.5
axs[1, 2].hist(RE_mlp_mse_x2, color='tab:red', edgecolor='black')
axs[1, 2].set_xlabel('MLP')
axs[1, 2].set_ylabel('Frequency')
# Plot 6
axs[2, 0].hist(RE_vae_mse_x2, color='tab:olive', edgecolor='black')
axs[2, 0].set_xlabel('VAE')
axs[2, 0].set_ylabel('Frequency')
# Plot 7
axs[2, 1].hist(RE_gae_mse_x2, color='tab:green', edgecolor='black')
axs[2, 1].set_xlabel('GAE')
axs[2, 1].set_ylabel('Frequency')
# Plot 8
axs[2, 2].hist(RE_dif_mse_x2, color='tab:brown', edgecolor='black')
axs[2, 2].set_xlabel('Diffusion')
axs[2, 2].set_ylabel('Frequency')

fig.suptitle('x2 MSE - Study Random Effects', fontsize=20)
#plt.savefig(filepath+"Figures/Case"+str(case)+"/mseX2_RE.png")
plt.savefig(filepath+"/mseX2_RE.png")
plt.close()

###

fig, axs = plt.subplots(3, 3, figsize=(12, 16))
plt.subplots_adjust(hspace=0.4, wspace=0.4)
# Plot 1
axs[0, 0].hist(RE_cca_mse_x3, color='tab:cyan', edgecolor='black')
axs[0, 0].set_xlabel('CCA')
axs[0, 0].set_ylabel('Frequency')
# Plot 2
axs[0, 1].hist(RE_mi_mse_x3, color='tab:blue', edgecolor='black')
axs[0, 1].set_xlabel('MI')
axs[0, 1].set_ylabel('Frequency')
# Plot 3
axs[0, 2].hist(RE_lh_mse_x3, color='tab:purple', edgecolor='black')
axs[0, 2].set_xlabel('Likelihood')
axs[0, 2].set_ylabel('Frequency')
# Plot 4
axs[1, 0].hist(RE_rf_mse_x3, color='tab:pink', edgecolor='black')
axs[1, 0].set_xlabel('Random Forest')
axs[1, 0].set_ylabel('Frequency')
# Plot 5
axs[1, 1].hist(RE_lgb_mse_x3, color='tab:orange', edgecolor='black')
axs[1, 1].set_xlabel('LGBM')
axs[1, 1].set_ylabel('Frequency')
# Plot 5.5
axs[1, 2].hist(RE_mlp_mse_x3, color='tab:red', edgecolor='black')
axs[1, 2].set_xlabel('MLP')
axs[1, 2].set_ylabel('Frequency')
# Plot 6
axs[2, 0].hist(RE_vae_mse_x3, color='tab:olive', edgecolor='black')
axs[2, 0].set_xlabel('VAE')
axs[2, 0].set_ylabel('Frequency')
# Plot 7
axs[2, 1].hist(RE_gae_mse_x3, color='tab:green', edgecolor='black')
axs[2, 1].set_xlabel('GAE')
axs[2, 1].set_ylabel('Frequency')
# Plot 8
axs[2, 2].hist(RE_dif_mse_x3, color='tab:brown', edgecolor='black')
axs[2, 2].set_xlabel('Diffusion')
axs[2, 2].set_ylabel('Frequency')

fig.suptitle('x3 MSE - Study Random Effects', fontsize=20)
#plt.savefig(filepath+"Figures/Case"+str(case)+"/mseX3_RE.png")
plt.savefig(filepath+"/mseX3_RE.png")
plt.close()



###### FEl ######
### BIAS ###

fig, axs = plt.subplots(3, 3, figsize=(12, 16))
plt.subplots_adjust(hspace=0.4, wspace=0.4)
# Plot 1
axs[0, 0].hist(FEt_cca_b0, color='tab:cyan', edgecolor='black')
#axs[0, 0].axvline(b0True, color='black', linewidth=3)
axs[0, 0].set_xlabel('CCA')
axs[0, 0].set_ylabel('Frequency')
# Plot 2
axs[0, 1].hist(FEt_mi_b0, color='tab:blue', edgecolor='black')
#axs[0, 1].axvline(b0True, color='black', linewidth=3)
axs[0, 1].set_xlabel('MI')
axs[0, 1].set_ylabel('Frequency')
# Plot 3
axs[0, 2].hist(FEt_lh_b0, color='tab:purple', edgecolor='black')
#axs[0, 2].axvline(b0True, color='black', linewidth=3)
axs[0, 2].set_xlabel('Likelihood')
axs[0, 2].set_ylabel('Frequency')
# Plot 4
axs[1, 0].hist(FEt_rf_b0, color='tab:pink', edgecolor='black')
#axs[1, 0].axvline(b0True, color='black', linewidth=3)
axs[1, 0].set_xlabel('Random Forest')
axs[1, 0].set_ylabel('Frequency')
# Plot 5
axs[1, 1].hist(FEt_lgb_b0, color='tab:orange', edgecolor='black')
#axs[1, 1].axvline(b0True, color='black', linewidth=3)
axs[1, 1].set_xlabel('LGBM')
axs[1, 1].set_ylabel('Frequency')
# Plot 5.5
axs[1, 2].hist(FEt_mlp_b0, color='tab:red', edgecolor='black')
#axs[1, 2].axvline(b0True, color='black', linewidth=3)
axs[1, 2].set_xlabel('MLP')
axs[1, 2].set_ylabel('Frequency')
# Plot 6
axs[2, 0].hist(FEt_vae_b0, color='tab:olive', edgecolor='black')
#axs[2, 0].axvline(b0True, color='black', linewidth=3)
axs[2, 0].set_xlabel('VAE')
axs[2, 0].set_ylabel('Frequency')
# Plot 7
axs[2, 1].hist(FEt_gae_b0, color='tab:green', edgecolor='black')
#axs[2, 1].axvline(b0True, color='black', linewidth=3)
axs[2, 1].set_xlabel('GAE')
axs[2, 1].set_ylabel('Frequency')
# Plot 8
axs[2, 2].hist(FEt_dif_b0, color='tab:brown', edgecolor='black')
#axs[2, 2].axvline(b0True, color='black', linewidth=3)
axs[2, 2].set_xlabel('Diffusion')
axs[2, 2].set_ylabel('Frequency')

fig.suptitle('Intercept Coefficient - Time Fixed Effects', fontsize=20)
#plt.savefig(filepath+"Figures/Case"+str(case)+"/BiasB0_FEt.png")
plt.savefig(filepath+"/BiasB0_FEt.png")
plt.close()

###

fig, axs = plt.subplots(3, 3, figsize=(12, 16))
plt.subplots_adjust(hspace=0.4, wspace=0.4)
# Plot 1
axs[0, 0].hist(FEt_cca_b1, color='tab:cyan', edgecolor='black')
axs[0, 0].axvline(b1True, color='black', linewidth=3)
axs[0, 0].set_xlabel('CCA')
axs[0, 0].set_ylabel('Frequency')
# Plot 2
axs[0, 1].hist(FEt_mi_b1, color='tab:blue', edgecolor='black')
axs[0, 1].axvline(b1True, color='black', linewidth=3)
axs[0, 1].set_xlabel('MI')
axs[0, 1].set_ylabel('Frequency')
# Plot 3
axs[0, 2].hist(FEt_lh_b1, color='tab:purple', edgecolor='black')
axs[0, 2].axvline(b1True, color='black', linewidth=3)
axs[0, 2].set_xlabel('Likelihood')
axs[0, 2].set_ylabel('Frequency')
# Plot 4
axs[1, 0].hist(FEt_rf_b1, color='tab:pink', edgecolor='black')
axs[1, 0].axvline(b1True, color='black', linewidth=3)
axs[1, 0].set_xlabel('Random Forest')
axs[1, 0].set_ylabel('Frequency')
# Plot 5
axs[1, 1].hist(FEt_lgb_b1, color='tab:orange', edgecolor='black')
axs[1, 1].axvline(b1True, color='black', linewidth=3)
axs[1, 1].set_xlabel('LGBM')
axs[1, 1].set_ylabel('Frequency')
# Plot 5.5
axs[1, 2].hist(FEt_mlp_b1, color='tab:red', edgecolor='black')
axs[1, 2].axvline(b1True, color='black', linewidth=3)
axs[1, 2].set_xlabel('MLP')
axs[1, 2].set_ylabel('Frequency')
# Plot 6
axs[2, 0].hist(FEt_vae_b1, color='tab:olive', edgecolor='black')
axs[2, 0].axvline(b1True, color='black', linewidth=3)
axs[2, 0].set_xlabel('VAE')
axs[2, 0].set_ylabel('Frequency')
# Plot 7
axs[2, 1].hist(FEt_gae_b1, color='tab:green', edgecolor='black')
axs[2, 1].axvline(b1True, color='black', linewidth=3)
axs[2, 1].set_xlabel('GAE')
axs[2, 1].set_ylabel('Frequency')
# Plot 8
axs[2, 2].hist(FEt_dif_b1, color='tab:brown', edgecolor='black')
axs[2, 2].axvline(b1True, color='black', linewidth=3)
axs[2, 2].set_xlabel('Diffusion')
axs[2, 2].set_ylabel('Frequency')

fig.suptitle('x1 Coefficient - Time Fixed Effects', fontsize=20)
#plt.savefig(filepath+"Figures/Case"+str(case)+"/BiasB1_FEt.png")
plt.savefig(filepath+"/BiasB1_FEt.png")
plt.close()

###

fig, axs = plt.subplots(3, 3, figsize=(12, 16))
plt.subplots_adjust(hspace=0.4, wspace=0.4)
# Plot 1
axs[0, 0].hist(FEt_cca_b2, color='tab:cyan', edgecolor='black')
axs[0, 0].axvline(b2True, color='black', linewidth=3)
axs[0, 0].set_xlabel('CCA')
axs[0, 0].set_ylabel('Frequency')
# Plot 2
axs[0, 1].hist(FEt_mi_b2, color='tab:blue', edgecolor='black')
axs[0, 1].axvline(b2True, color='black', linewidth=3)
axs[0, 1].set_xlabel('MI')
axs[0, 1].set_ylabel('Frequency')
# Plot 3
axs[0, 2].hist(FEt_lh_b2, color='tab:purple', edgecolor='black')
axs[0, 2].axvline(b2True, color='black', linewidth=3)
axs[0, 2].set_xlabel('Likelihood')
axs[0, 2].set_ylabel('Frequency')
# Plot 4
axs[1, 0].hist(FEt_rf_b2, color='tab:pink', edgecolor='black')
axs[1, 0].axvline(b2True, color='black', linewidth=3)
axs[1, 0].set_xlabel('Random Forest')
axs[1, 0].set_ylabel('Frequency')
# Plot 5
axs[1, 1].hist(FEt_lgb_b2, color='tab:orange', edgecolor='black')
axs[1, 1].axvline(b2True, color='black', linewidth=3)
axs[1, 1].set_xlabel('LGBM')
axs[1, 1].set_ylabel('Frequency')
# Plot 5.5
axs[1, 2].hist(FEt_mlp_b2, color='tab:red', edgecolor='black')
axs[1, 2].axvline(b2True, color='black', linewidth=3)
axs[1, 2].set_xlabel('MLP')
axs[1, 2].set_ylabel('Frequency')
# Plot 6
axs[2, 0].hist(FEt_vae_b2, color='tab:olive', edgecolor='black')
axs[2, 0].axvline(b2True, color='black', linewidth=3)
axs[2, 0].set_xlabel('VAE')
axs[2, 0].set_ylabel('Frequency')
# Plot 7
axs[2, 1].hist(FEt_gae_b2, color='tab:green', edgecolor='black')
axs[2, 1].axvline(b2True, color='black', linewidth=3)
axs[2, 1].set_xlabel('GAE')
axs[2, 1].set_ylabel('Frequency')
# Plot 8
axs[2, 2].hist(FEt_dif_b2, color='tab:brown', edgecolor='black')
axs[2, 2].axvline(b2True, color='black', linewidth=3)
axs[2, 2].set_xlabel('Diffusion')
axs[2, 2].set_ylabel('Frequency')

fig.suptitle('x2 Coefficient - Time Fixed Effects', fontsize=20)
#plt.savefig(filepath+"Figures/Case"+str(case)+"/BiasB2_FEt.png")
plt.savefig(filepath+"/BiasB2_FEt.png")
plt.close()

###

fig, axs = plt.subplots(3, 3, figsize=(12, 16))
plt.subplots_adjust(hspace=0.4, wspace=0.4)
# Plot 1
axs[0, 0].hist(FEt_cca_b3, color='tab:cyan', edgecolor='black')
axs[0, 0].axvline(b3True, color='black', linewidth=3)
axs[0, 0].set_xlabel('CCA')
axs[0, 0].set_ylabel('Frequency')
# Plot 2
axs[0, 1].hist(FEt_mi_b3, color='tab:blue', edgecolor='black')
axs[0, 1].axvline(b3True, color='black', linewidth=3)
axs[0, 1].set_xlabel('MI')
axs[0, 1].set_ylabel('Frequency')
# Plot 3
axs[0, 2].hist(FEt_lh_b3, color='tab:purple', edgecolor='black')
axs[0, 2].axvline(b3True, color='black', linewidth=3)
axs[0, 2].set_xlabel('Likelihood')
axs[0, 2].set_ylabel('Frequency')
# Plot 4
axs[1, 0].hist(FEt_rf_b3, color='tab:pink', edgecolor='black')
axs[1, 0].axvline(b3True, color='black', linewidth=3)
axs[1, 0].set_xlabel('Random Forest')
axs[1, 0].set_ylabel('Frequency')
# Plot 5
axs[1, 1].hist(FEt_lgb_b3, color='tab:orange', edgecolor='black')
axs[1, 1].axvline(b3True, color='black', linewidth=3)
axs[1, 1].set_xlabel('LGBM')
axs[1, 1].set_ylabel('Frequency')
# Plot 5.5
axs[1, 2].hist(FEt_mlp_b3, color='tab:red', edgecolor='black')
axs[1, 2].axvline(b3True, color='black', linewidth=3)
axs[1, 2].set_xlabel('MLP')
axs[1, 2].set_ylabel('Frequency')
# Plot 6
axs[2, 0].hist(FEt_vae_b3, color='tab:olive', edgecolor='black')
axs[2, 0].axvline(b3True, color='black', linewidth=3)
axs[2, 0].set_xlabel('VAE')
axs[2, 0].set_ylabel('Frequency')
# Plot 7
axs[2, 1].hist(FEt_gae_b3, color='tab:green', edgecolor='black')
axs[2, 1].axvline(b3True, color='black', linewidth=3)
axs[2, 1].set_xlabel('GAE')
axs[2, 1].set_ylabel('Frequency')
# Plot 8
axs[2, 2].hist(FEt_dif_b3, color='tab:brown', edgecolor='black')
axs[2, 2].axvline(b3True, color='black', linewidth=3)
axs[2, 2].set_xlabel('Diffusion')
axs[2, 2].set_ylabel('Frequency')

fig.suptitle('x3 Coefficient - Time Fixed Effects', fontsize=20)
#plt.savefig(filepath+"Figures/Case"+str(case)+"/BiasB3_FEt.png")
plt.savefig(filepath+"/BiasB3_FEt.png")
plt.close()

### VARIANCE ###

fig, axs = plt.subplots(3, 3, figsize=(12, 16))
plt.subplots_adjust(hspace=0.4, wspace=0.4)
# Plot 1
axs[0, 0].hist(FEt_cca_b1Var, color='tab:cyan', edgecolor='black')
axs[0, 0].set_xlabel('CCA')
axs[0, 0].set_ylabel('Frequency')
# Plot 2
axs[0, 1].hist(FEt_mi_b1Var, color='tab:blue', edgecolor='black')
axs[0, 1].set_xlabel('MI')
axs[0, 1].set_ylabel('Frequency')
# Plot 3
axs[0, 2].hist(FEt_lh_b1Var, color='tab:purple', edgecolor='black')
axs[0, 2].set_xlabel('Likelihood')
axs[0, 2].set_ylabel('Frequency')
# Plot 4
axs[1, 0].hist(FEt_rf_b1Var, color='tab:pink', edgecolor='black')
axs[1, 0].set_xlabel('Random Forest')
axs[1, 0].set_ylabel('Frequency')
# Plot 5
axs[1, 1].hist(FEt_lgb_b1Var, color='tab:orange', edgecolor='black')
axs[1, 1].set_xlabel('LGBM')
axs[1, 1].set_ylabel('Frequency')
# Plot 5.5
axs[1, 2].hist(FEt_mlp_b1Var, color='tab:red', edgecolor='black')
axs[1, 2].set_xlabel('MLP')
axs[1, 2].set_ylabel('Frequency')
# Plot 6
axs[2, 0].hist(FEt_vae_b1Var, color='tab:olive', edgecolor='black')
axs[2, 0].set_xlabel('VAE')
axs[2, 0].set_ylabel('Frequency')
# Plot 7
axs[2, 1].hist(FEt_gae_b1Var, color='tab:green', edgecolor='black')
axs[2, 1].set_xlabel('GAE')
axs[2, 1].set_ylabel('Frequency')
# Plot 8
axs[2, 2].hist(FEt_dif_b1Var, color='tab:brown', edgecolor='black')
axs[2, 2].set_xlabel('Diffusion')
axs[2, 2].set_ylabel('Frequency')

fig.suptitle('x1 Variance - Time Fixed Effects', fontsize=20)
#plt.savefig(filepath+"Figures/Case"+str(case)+"/VarB1_FEt.png")
plt.savefig(filepath+"/VarB1_FEt.png")
plt.close()

###

fig, axs = plt.subplots(3, 3, figsize=(12, 16))
plt.subplots_adjust(hspace=0.4, wspace=0.4)
# Plot 1
axs[0, 0].hist(FEt_cca_b2Var, color='tab:cyan', edgecolor='black')
axs[0, 0].set_xlabel('CCA')
axs[0, 0].set_ylabel('Frequency')
# Plot 2
axs[0, 1].hist(FEt_mi_b2Var, color='tab:blue', edgecolor='black')
axs[0, 1].set_xlabel('MI')
axs[0, 1].set_ylabel('Frequency')
# Plot 3
axs[0, 2].hist(FEt_lh_b2Var, color='tab:purple', edgecolor='black')
axs[0, 2].set_xlabel('Likelihood')
axs[0, 2].set_ylabel('Frequency')
# Plot 4
axs[1, 0].hist(FEt_rf_b2Var, color='tab:pink', edgecolor='black')
axs[1, 0].set_xlabel('Random Forest')
axs[1, 0].set_ylabel('Frequency')
# Plot 5
axs[1, 1].hist(FEt_lgb_b2Var, color='tab:orange', edgecolor='black')
axs[1, 1].set_xlabel('LGBM')
axs[1, 1].set_ylabel('Frequency')
# Plot 5.5
axs[1, 2].hist(FEt_mlp_b2Var, color='tab:red', edgecolor='black')
axs[1, 2].set_xlabel('MLP')
axs[1, 2].set_ylabel('Frequency')
# Plot 6
axs[2, 0].hist(FEt_vae_b2Var, color='tab:olive', edgecolor='black')
axs[2, 0].set_xlabel('VAE')
axs[2, 0].set_ylabel('Frequency')
# Plot 7
axs[2, 1].hist(FEt_gae_b2Var, color='tab:green', edgecolor='black')
axs[2, 1].set_xlabel('GAE')
axs[2, 1].set_ylabel('Frequency')
# Plot 8
axs[2, 2].hist(FEt_dif_b2Var, color='tab:brown', edgecolor='black')
axs[2, 2].set_xlabel('Diffusion')
axs[2, 2].set_ylabel('Frequency')

fig.suptitle('x2 Variance - Time Fixed Effects', fontsize=20)
#plt.savefig(filepath+"Figures/Case"+str(case)+"/VarB2_FEt.png")
plt.savefig(filepath+"/VarB2_FEt.png")
plt.close()

###

fig, axs = plt.subplots(3, 3, figsize=(12, 16))
plt.subplots_adjust(hspace=0.4, wspace=0.4)
# Plot 1
axs[0, 0].hist(FEt_cca_b3Var, color='tab:cyan', edgecolor='black')
axs[0, 0].set_xlabel('CCA')
axs[0, 0].set_ylabel('Frequency')
# Plot 2
axs[0, 1].hist(FEt_mi_b3Var, color='tab:blue', edgecolor='black')
axs[0, 1].set_xlabel('MI')
axs[0, 1].set_ylabel('Frequency')
# Plot 3
axs[0, 2].hist(FEt_lh_b3Var, color='tab:purple', edgecolor='black')
axs[0, 2].set_xlabel('Likelihood')
axs[0, 2].set_ylabel('Frequency')
# Plot 4
axs[1, 0].hist(FEt_rf_b3Var, color='tab:pink', edgecolor='black')
axs[1, 0].set_xlabel('Random Forest')
axs[1, 0].set_ylabel('Frequency')
# Plot 5
axs[1, 1].hist(FEt_lgb_b3Var, color='tab:orange', edgecolor='black')
axs[1, 1].set_xlabel('LGBM')
axs[1, 1].set_ylabel('Frequency')
# Plot 5.5
axs[1, 2].hist(FEt_mlp_b3Var, color='tab:red', edgecolor='black')
axs[1, 2].set_xlabel('MLP')
axs[1, 2].set_ylabel('Frequency')
# Plot 6
axs[2, 0].hist(FEt_vae_b3Var, color='tab:olive', edgecolor='black')
axs[2, 0].set_xlabel('VAE')
axs[2, 0].set_ylabel('Frequency')
# Plot 7
axs[2, 1].hist(FEt_gae_b3Var, color='tab:green', edgecolor='black')
axs[2, 1].set_xlabel('GAE')
axs[2, 1].set_ylabel('Frequency')
# Plot 8
axs[2, 2].hist(FEt_dif_b3Var, color='tab:brown', edgecolor='black')
axs[2, 2].set_xlabel('Diffusion')
axs[2, 2].set_ylabel('Frequency')

fig.suptitle('x3 Variance - Time Fixed Effects', fontsize=20)
#plt.savefig(filepath+"Figures/Case"+str(case)+"/VarB3_FEt.png")
plt.savefig(filepath+"/VarB3_FEt.png")
plt.close()

### MSE ###

fig, axs = plt.subplots(3, 3, figsize=(12, 16))
plt.subplots_adjust(hspace=0.4, wspace=0.4)
# Plot 1
axs[0, 0].hist(FEt_cca_mse, color='tab:cyan', edgecolor='black')
axs[0, 0].set_xlabel('CCA')
axs[0, 0].set_ylabel('Frequency')
# Plot 2
axs[0, 1].hist(FEt_mi_mse, color='tab:blue', edgecolor='black')
axs[0, 1].set_xlabel('MI')
axs[0, 1].set_ylabel('Frequency')
# Plot 3
axs[0, 2].hist(FEt_lh_mse, color='tab:purple', edgecolor='black')
axs[0, 2].set_xlabel('Likelihood')
axs[0, 2].set_ylabel('Frequency')
# Plot 4
axs[1, 0].hist(FEt_rf_mse, color='tab:pink', edgecolor='black')
axs[1, 0].set_xlabel('Random Forest')
axs[1, 0].set_ylabel('Frequency')
# Plot 5
axs[1, 1].hist(FEt_lgb_mse, color='tab:orange', edgecolor='black')
axs[1, 1].set_xlabel('LGBM')
axs[1, 1].set_ylabel('Frequency')
# Plot 5.5
axs[1, 2].hist(FEt_mlp_mse, color='tab:red', edgecolor='black')
axs[1, 2].set_xlabel('MLP')
axs[1, 2].set_ylabel('Frequency')
# Plot 6
axs[2, 0].hist(FEt_vae_mse, color='tab:olive', edgecolor='black')
axs[2, 0].set_xlabel('VAE')
axs[2, 0].set_ylabel('Frequency')
# Plot 7
axs[2, 1].hist(FEt_gae_mse, color='tab:green', edgecolor='black')
axs[2, 1].set_xlabel('GAE')
axs[2, 1].set_ylabel('Frequency')
# Plot 8
axs[2, 2].hist(FEt_dif_mse, color='tab:brown', edgecolor='black')
axs[2, 2].set_xlabel('Diffusion')
axs[2, 2].set_ylabel('Frequency')

fig.suptitle('y MSE - Time Fixed Effects', fontsize=20)
#plt.savefig(filepath+"Figures/Case"+str(case)+"/mse_FEt.png")
plt.savefig(filepath+"/mse_FEt.png")
plt.close()

###

fig, axs = plt.subplots(3, 3, figsize=(12, 16))
plt.subplots_adjust(hspace=0.4, wspace=0.4)
# Plot 1
axs[0, 0].hist(FEt_cca_mse_x1, color='tab:cyan', edgecolor='black')
axs[0, 0].set_xlabel('CCA')
axs[0, 0].set_ylabel('Frequency')
# Plot 2
axs[0, 1].hist(FEt_mi_mse_x1, color='tab:blue', edgecolor='black')
axs[0, 1].set_xlabel('MI')
axs[0, 1].set_ylabel('Frequency')
# Plot 3
axs[0, 2].hist(FEt_lh_mse_x1, color='tab:purple', edgecolor='black')
axs[0, 2].set_xlabel('Likelihood')
axs[0, 2].set_ylabel('Frequency')
# Plot 4
axs[1, 0].hist(FEt_rf_mse_x1, color='tab:pink', edgecolor='black')
axs[1, 0].set_xlabel('Random Forest')
axs[1, 0].set_ylabel('Frequency')
# Plot 5
axs[1, 1].hist(FEt_lgb_mse_x1, color='tab:orange', edgecolor='black')
axs[1, 1].set_xlabel('LGBM')
axs[1, 1].set_ylabel('Frequency')
# Plot 5.5
axs[1, 2].hist(FEt_mlp_mse_x1, color='tab:red', edgecolor='black')
axs[1, 2].set_xlabel('MLP')
axs[1, 2].set_ylabel('Frequency')
# Plot 6
axs[2, 0].hist(FEt_vae_mse_x1, color='tab:olive', edgecolor='black')
axs[2, 0].set_xlabel('VAE')
axs[2, 0].set_ylabel('Frequency')
# Plot 7
axs[2, 1].hist(FEt_gae_mse_x1, color='tab:green', edgecolor='black')
axs[2, 1].set_xlabel('GAE')
axs[2, 1].set_ylabel('Frequency')
# Plot 8
axs[2, 2].hist(FEt_dif_mse_x1, color='tab:brown', edgecolor='black')
axs[2, 2].set_xlabel('Diffusion')
axs[2, 2].set_ylabel('Frequency')

fig.suptitle('x1 MSE - Time Fixed Effects', fontsize=20)
#plt.savefig(filepath+"Figures/Case"+str(case)+"/mseX1_FEt.png")
plt.savefig(filepath+"/mseX1_FEt.png")
plt.close()

###

fig, axs = plt.subplots(3, 3, figsize=(12, 16))
plt.subplots_adjust(hspace=0.4, wspace=0.4)
# Plot 1
axs[0, 0].hist(FEt_cca_mse_x2, color='tab:cyan', edgecolor='black')
axs[0, 0].set_xlabel('CCA')
axs[0, 0].set_ylabel('Frequency')
# Plot 2
axs[0, 1].hist(FEt_mi_mse_x2, color='tab:blue', edgecolor='black')
axs[0, 1].set_xlabel('MI')
axs[0, 1].set_ylabel('Frequency')
# Plot 3
axs[0, 2].hist(FEt_lh_mse_x2, color='tab:purple', edgecolor='black')
axs[0, 2].set_xlabel('Likelihood')
axs[0, 2].set_ylabel('Frequency')
# Plot 4
axs[1, 0].hist(FEt_rf_mse_x2, color='tab:pink', edgecolor='black')
axs[1, 0].set_xlabel('Random Forest')
axs[1, 0].set_ylabel('Frequency')
# Plot 5
axs[1, 1].hist(FEt_lgb_mse_x2, color='tab:orange', edgecolor='black')
axs[1, 1].set_xlabel('LGBM')
axs[1, 1].set_ylabel('Frequency')
# Plot 5.5
axs[1, 2].hist(FEt_mlp_mse_x2, color='tab:red', edgecolor='black')
axs[1, 2].set_xlabel('MLP')
axs[1, 2].set_ylabel('Frequency')
# Plot 6
axs[2, 0].hist(FEt_vae_mse_x2, color='tab:olive', edgecolor='black')
axs[2, 0].set_xlabel('VAE')
axs[2, 0].set_ylabel('Frequency')
# Plot 7
axs[2, 1].hist(FEt_gae_mse_x2, color='tab:green', edgecolor='black')
axs[2, 1].set_xlabel('GAE')
axs[2, 1].set_ylabel('Frequency')
# Plot 8
axs[2, 2].hist(FEt_dif_mse_x2, color='tab:brown', edgecolor='black')
axs[2, 2].set_xlabel('Diffusion')
axs[2, 2].set_ylabel('Frequency')

fig.suptitle('x2 MSE - Time Fixed Effects', fontsize=20)
#plt.savefig(filepath+"Figures/Case"+str(case)+"/mseX2_FEt.png")
plt.savefig(filepath+"/mseX2_FEt.png")
plt.close()

###

fig, axs = plt.subplots(3, 3, figsize=(12, 16))
plt.subplots_adjust(hspace=0.4, wspace=0.4)
# Plot 1
axs[0, 0].hist(FEt_cca_mse_x3, color='tab:cyan', edgecolor='black')
axs[0, 0].set_xlabel('CCA')
axs[0, 0].set_ylabel('Frequency')
# Plot 2
axs[0, 1].hist(FEt_mi_mse_x3, color='tab:blue', edgecolor='black')
axs[0, 1].set_xlabel('MI')
axs[0, 1].set_ylabel('Frequency')
# Plot 3
axs[0, 2].hist(FEt_lh_mse_x3, color='tab:purple', edgecolor='black')
axs[0, 2].set_xlabel('Likelihood')
axs[0, 2].set_ylabel('Frequency')
# Plot 4
axs[1, 0].hist(FEt_rf_mse_x3, color='tab:pink', edgecolor='black')
axs[1, 0].set_xlabel('Random Forest')
axs[1, 0].set_ylabel('Frequency')
# Plot 5
axs[1, 1].hist(FEt_lgb_mse_x3, color='tab:orange', edgecolor='black')
axs[1, 1].set_xlabel('LGBM')
axs[1, 1].set_ylabel('Frequency')
# Plot 5.5
axs[1, 2].hist(FEt_mlp_mse_x3, color='tab:red', edgecolor='black')
axs[1, 2].set_xlabel('MLP')
axs[1, 2].set_ylabel('Frequency')
# Plot 6
axs[2, 0].hist(FEt_vae_mse_x3, color='tab:olive', edgecolor='black')
axs[2, 0].set_xlabel('VAE')
axs[2, 0].set_ylabel('Frequency')
# Plot 7
axs[2, 1].hist(FEt_gae_mse_x3, color='tab:green', edgecolor='black')
axs[2, 1].set_xlabel('GAE')
axs[2, 1].set_ylabel('Frequency')
# Plot 8
axs[2, 2].hist(FEt_dif_mse_x3, color='tab:brown', edgecolor='black')
axs[2, 2].set_xlabel('Diffusion')
axs[2, 2].set_ylabel('Frequency')

fig.suptitle('x3 MSE - Time Fixed Effects', fontsize=20)
#plt.savefig(filepath+"Figures/Case"+str(case)+"/mseX3_FEt.png")
plt.savefig(filepath+"/mseX3_FEt.png")
plt.close()



###############################################################################
###                                 POWER                                   ###
###############################################################################

alphaLevel = 0.000001 # for N = 7,500, adjusted from 0.000005

#discrep_interval = 0.1
#
discrep_interval = 0.005

from scipy.stats import norm
def power_curve(trueCoefs, SE, discrep_range, discrep_interval, df,
                alphaLevel, subtitle, yTitle, ax):
    
    trueCoefs = np.atleast_1d(trueCoefs).flatten()
    SE = np.atleast_1d(SE).flatten()
    crit_value = t.ppf(1 - alphaLevel / 2, df)
    
    max_discrep = discrep_range * discrep_interval
    # 1000 points for a high-resolution, silky smooth line
    x_steps = np.linspace(-max_discrep, max_discrep, 1000)
    
    power_out = []

    for d in x_steps:
        pow_vals = []
        for j in range(len(trueCoefs)):
            delta = abs(d) / SE[j]
            
            # --- THE NATURAL CURVE FIX ---
            # We use the Normal Approximation (limit of t-dist) 
            # This is mathematically smooth and prevents 'squaring off'
            # Power = P(Z > crit - delta) + P(Z < -crit - delta)
            val = norm.sf(crit_value - delta) + norm.cdf(-crit_value - delta)
            
            pow_vals.append(val)
        
        power_out.append(np.mean(pow_vals))

    # Convert to array and ensure it's mathematically bounded
    power_out = np.clip(np.array(power_out), 0, 1)

    # Plotting with the same visual style
    ax.plot(x_steps, power_out, 
            linestyle=(0, (5, 2)), 
            color='red', 
            linewidth=2, 
            label='Power', 
            zorder=10)

    # Standard formatting
    ax.set_xlabel("Discrepancy")
    ax.set_ylabel(yTitle)
    ax.set_title(subtitle)
    ax.legend(loc='lower right', fontsize='small')
    ax.grid(True, linestyle=':', alpha=0.4, zorder=1)
    
    ax.set_xlim(-max_discrep, max_discrep)
    ax.set_ylim(-0.02, 1.05)

##############################
###          X1            ###
##############################

###### FEs ######

# Set up the figure with 3 rows and 4 columns of subplots
fig, axs = plt.subplots(3, 3, figsize=(12, 16))
fig.suptitle('Power of X1 (Study Fixed Effects)', fontsize=16, fontweight='bold')
axs = axs.flatten()
#plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.1, hspace=0.4, wspace=0.4)

# Assuming power_curve is a function that plots on a given axis
# You need to define power_curve to accept an axis parameter for plotting

power_curve(trueBetas[0], FE_cca_b1SE, 100, discrep_interval, FE_cca_.df_resid, alphaLevel, "CCA", "X1", axs[0])
power_curve(trueBetas[0], FE_mi_b1SE, 100, discrep_interval, FE_mi_.df_resid, alphaLevel, "Multiple Imputation", "X1", axs[1])
power_curve(trueBetas[0], FE_lh_b1SE, 100, discrep_interval, FE_lh_.df_resid, alphaLevel, "Likelihood", "X1", axs[2])
power_curve(trueBetas[0], FE_rf_b1SE, 100, discrep_interval, FE_rf_.df_resid, alphaLevel, "Random Forest", "X1", axs[3])
###
power_curve(trueBetas[0], FE_lgb_b1SE, 100, discrep_interval, FE_lgb_.df_resid, alphaLevel, "LGBM", "X1", axs[4])
power_curve(trueBetas[0], FE_mlp_b1SE, 100, discrep_interval, FE_mlp_.df_resid, alphaLevel, "MLP", "X1", axs[5])
power_curve(trueBetas[0], FE_vae_b1SE, 100, discrep_interval, FE_vae_.df_resid, alphaLevel, "VAE", "X1", axs[6])
power_curve(trueBetas[0], FE_gae_b1SE, 100, discrep_interval, FE_gae_.df_resid, alphaLevel, "GAE", "X1", axs[7])
power_curve(trueBetas[0], FE_dif_b1SE, 100, discrep_interval, FE_dif_.df_resid, alphaLevel, "Diffusion", "X1", axs[8])

# Save the figure to a file
#plt.savefig(filepath+"Figures/Case"+str(case)+"/powerX1_FEs.png")
plt.savefig(filepath+"/powerX1_FEs.png")
plt.close(fig)

###### REl ######

# Set up the figure with 3 rows and 4 columns of subplots
fig, axs = plt.subplots(3, 3, figsize=(12, 16))
fig.suptitle('Power of X1 (Study Random Effects)', fontsize=16, fontweight='bold')
axs = axs.flatten()
#plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.1, hspace=0.4, wspace=0.4)

# Assuming power_curve is a function that plots on a given axis
# You need to define power_curve to accept an axis parameter for plotting

power_curve(trueBetas[0], RE_cca_b1SE, 100, discrep_interval, RE_cca_.df_resid, alphaLevel, "CCA", "X1", axs[0])
power_curve(trueBetas[0], RE_mi_b1SE, 100, discrep_interval, RE_mi_.df_resid, alphaLevel, "Multiple Imputation", "X1", axs[1])
power_curve(trueBetas[0], RE_lh_b1SE, 100, discrep_interval, RE_lh_.df_resid, alphaLevel, "Likelihood", "X1", axs[2])
power_curve(trueBetas[0], RE_rf_b1SE, 100, discrep_interval, RE_rf_.df_resid, alphaLevel, "Random Forest", "X1", axs[3])
###
power_curve(trueBetas[0], RE_lgb_b1SE, 100, discrep_interval, RE_lgb_.df_resid, alphaLevel, "LGBM", "X1", axs[4])
power_curve(trueBetas[0], RE_mlp_b1SE, 100, discrep_interval, RE_mlp_.df_resid, alphaLevel, "MLP", "X1", axs[5])
power_curve(trueBetas[0], RE_vae_b1SE, 100, discrep_interval, RE_vae_.df_resid, alphaLevel, "VAE", "X1", axs[6])
power_curve(trueBetas[0], RE_gae_b1SE, 100, discrep_interval, RE_gae_.df_resid, alphaLevel, "GAE", "X1", axs[7])
power_curve(trueBetas[0], RE_dif_b1SE, 100, discrep_interval, RE_dif_.df_resid, alphaLevel, "Diffusion", "X1", axs[8])

# Save the figure to a file
#plt.savefig(filepath+"Figures/Case"+str(case)+"/powerX1_RE.png")
plt.savefig(filepath+"/powerX1_RE.png")
plt.close(fig)

###### FEl ######

# Set up the figure with 3 rows and 4 columns of subplots
fig, axs = plt.subplots(3, 3, figsize=(12, 16))
fig.suptitle('Power of X1 (Time Fixed Effects)', fontsize=16, fontweight='bold')
axs = axs.flatten()
#plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.1, hspace=0.4, wspace=0.4)

# Assuming power_curve is a function that plots on a given axis
# You need to define power_curve to accept an axis parameter for plotting

power_curve(trueBetas[0], FEt_cca_b1SE, 100, discrep_interval, FEt_cca_.df_resid, alphaLevel, "CCA", "X1", axs[0])
power_curve(trueBetas[0], FEt_mi_b1SE, 100, discrep_interval, FEt_mi_.df_resid, alphaLevel, "Multiple Imputation", "X1", axs[1])
power_curve(trueBetas[0], FEt_lh_b1SE, 100, discrep_interval, FEt_lh_.df_resid, alphaLevel, "Likelihood", "X1", axs[2])
power_curve(trueBetas[0], FEt_rf_b1SE, 100, discrep_interval, FEt_rf_.df_resid, alphaLevel, "Random Forest", "X1", axs[3])
###
power_curve(trueBetas[0], FEt_lgb_b1SE, 100, discrep_interval, FEt_lgb_.df_resid, alphaLevel, "LGBM", "X1", axs[4])
power_curve(trueBetas[0], FEt_mlp_b1SE, 100, discrep_interval, FEt_mlp_.df_resid, alphaLevel, "MLP", "X1", axs[5])
power_curve(trueBetas[0], FEt_vae_b1SE, 100, discrep_interval, FEt_vae_.df_resid, alphaLevel, "VAE", "X1", axs[6])
power_curve(trueBetas[0], FEt_gae_b1SE, 100, discrep_interval, FEt_gae_.df_resid, alphaLevel, "GAE", "X1", axs[7])
power_curve(trueBetas[0], FEt_dif_b1SE, 100, discrep_interval, FEt_dif_.df_resid, alphaLevel, "Diffusion", "X1", axs[8])

# Save the figure to a file
#plt.savefig(filepath+"Figures/Case"+str(case)+"/powerX1_FEt.png")
plt.savefig(filepath+"/powerX1_FEt.png")
plt.close(fig)

##############################
###          X2            ###
##############################

###### FEs ######

# Set up the figure with 3 rows and 4 columns of subplots
fig, axs = plt.subplots(3, 3, figsize=(12, 16))
fig.suptitle('Power of X2 (Study Fixed Effects)', fontsize=16, fontweight='bold')
axs = axs.flatten()
#plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.1, hspace=0.4, wspace=0.4)

# Assuming power_curve is a function that plots on a given axis
# You need to define power_curve to accept an axis parameter for plotting

power_curve(trueBetas[1], FE_cca_b2SE, 100, discrep_interval, FE_cca_.df_resid, alphaLevel, "CCA", "X2", axs[0])
power_curve(trueBetas[1], FE_mi_b2SE, 100, discrep_interval, FE_mi_.df_resid, alphaLevel, "Multiple Imputation", "X2", axs[1])
power_curve(trueBetas[1], FE_lh_b2SE, 100, discrep_interval, FE_lh_.df_resid, alphaLevel, "Likelihood", "X2", axs[2])
power_curve(trueBetas[1], FE_rf_b2SE, 100, discrep_interval, FE_rf_.df_resid, alphaLevel, "Random Forest", "X2", axs[3])
###
power_curve(trueBetas[1], FE_lgb_b2SE, 100, discrep_interval, FE_lgb_.df_resid, alphaLevel, "LGBM", "X2", axs[4])
power_curve(trueBetas[1], FE_mlp_b2SE, 100, discrep_interval, FE_mlp_.df_resid, alphaLevel, "MLP", "X2", axs[5])
power_curve(trueBetas[1], FE_vae_b2SE, 100, discrep_interval, FE_vae_.df_resid, alphaLevel, "VAE", "X2", axs[6])
power_curve(trueBetas[1], FE_gae_b2SE, 100, discrep_interval, FE_gae_.df_resid, alphaLevel, "GAE", "X2", axs[7])
power_curve(trueBetas[1], FE_dif_b2SE, 100, discrep_interval, FE_dif_.df_resid, alphaLevel, "Diffusion", "X2", axs[8])

# Save the figure to a file
#plt.savefig(filepath+"Figures/Case"+str(case)+"/powerX2_FEs.png")
plt.savefig(filepath+"/powerX2_FEs.png")
plt.close(fig)

###### REl ######

# Set up the figure with 3 rows and 4 columns of subplots
fig, axs = plt.subplots(3, 3, figsize=(12, 16))
fig.suptitle('Power of X2 (Study Random Effects)', fontsize=16, fontweight='bold')
axs = axs.flatten()
#plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.1, hspace=0.4, wspace=0.4)

# Assuming power_curve is a function that plots on a given axis
# You need to define power_curve to accept an axis parameter for plotting

power_curve(trueBetas[1], RE_cca_b2SE, 100, discrep_interval, RE_cca_.df_resid, alphaLevel, "CCA", "X2", axs[0])
power_curve(trueBetas[1], RE_mi_b2SE, 100, discrep_interval, RE_mi_.df_resid, alphaLevel, "Multiple Imputation", "X2", axs[1])
power_curve(trueBetas[1], RE_lh_b2SE, 100, discrep_interval, RE_lh_.df_resid, alphaLevel, "Likelihood", "X2", axs[2])
power_curve(trueBetas[1], RE_rf_b2SE, 100, discrep_interval, RE_rf_.df_resid, alphaLevel, "Random Forest", "X2", axs[3])
###
power_curve(trueBetas[1], RE_lgb_b2SE, 100, discrep_interval, RE_lgb_.df_resid, alphaLevel, "LGBM", "X2", axs[4])
power_curve(trueBetas[1], RE_mlp_b2SE, 100, discrep_interval, RE_mlp_.df_resid, alphaLevel, "MLP", "X2", axs[5])
power_curve(trueBetas[1], RE_vae_b2SE, 100, discrep_interval, RE_vae_.df_resid, alphaLevel, "VAE", "X2", axs[6])
power_curve(trueBetas[1], RE_gae_b2SE, 100, discrep_interval, RE_gae_.df_resid, alphaLevel, "GAE", "X2", axs[7])
power_curve(trueBetas[1], RE_dif_b2SE, 100, discrep_interval, RE_dif_.df_resid, alphaLevel, "Diffusion", "X2", axs[8])

# Save the figure to a file
#plt.savefig(filepath+"Figures/Case"+str(case)+"/powerX2_RE.png")
plt.savefig(filepath+"/powerX2_RE.png")
plt.close(fig)

###### FEl ######

# Set up the figure with 3 rows and 4 columns of subplots
fig, axs = plt.subplots(3, 3, figsize=(12, 16))
fig.suptitle('Power of X2 (Time Fixed Effects)', fontsize=16, fontweight='bold')
axs = axs.flatten()
#plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.1, hspace=0.4, wspace=0.4)

# Assuming power_curve is a function that plots on a given axis
# You need to define power_curve to accept an axis parameter for plotting

power_curve(trueBetas[1], FEt_cca_b2SE, 100, discrep_interval, FEt_cca_.df_resid, alphaLevel, "CCA", "X2", axs[0])
power_curve(trueBetas[1], FEt_mi_b2SE, 100, discrep_interval, FEt_mi_.df_resid, alphaLevel, "Multiple Imputation", "X2", axs[1])
power_curve(trueBetas[1], FEt_lh_b2SE, 100, discrep_interval, FEt_lh_.df_resid, alphaLevel, "Likelihood", "X2", axs[2])
power_curve(trueBetas[1], FEt_rf_b2SE, 100, discrep_interval, FEt_rf_.df_resid, alphaLevel, "Random Forest", "X2", axs[3])
###
power_curve(trueBetas[1], FEt_lgb_b2SE, 100, discrep_interval, FEt_lgb_.df_resid, alphaLevel, "LGBM", "X2", axs[4])
power_curve(trueBetas[1], FEt_mlp_b2SE, 100, discrep_interval, FEt_mlp_.df_resid, alphaLevel, "MLP", "X2", axs[5])
power_curve(trueBetas[1], FEt_vae_b2SE, 100, discrep_interval, FEt_vae_.df_resid, alphaLevel, "VAE", "X2", axs[6])
power_curve(trueBetas[1], FEt_gae_b2SE, 100, discrep_interval, FEt_gae_.df_resid, alphaLevel, "GAE", "X2", axs[7])
power_curve(trueBetas[1], FEt_dif_b2SE, 100, discrep_interval, FEt_dif_.df_resid, alphaLevel, "Diffusion", "X2", axs[8])

# Save the figure to a file
#plt.savefig(filepath+"Figures/Case"+str(case)+"/powerX2_FEt.png")
plt.savefig(filepath+"/powerX2_FEt.png")
plt.close(fig)

##############################
###          X3            ###
##############################

###### FEs ######

# Set up the figure with 3 rows and 4 columns of subplots
fig, axs = plt.subplots(3, 3, figsize=(12, 16))
fig.suptitle('Power of X3 (Study Fixed Effects)', fontsize=16, fontweight='bold')
axs = axs.flatten()
#plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.1, hspace=0.4, wspace=0.4)

# Assuming power_curve is a function that plots on a given axis
# You need to define power_curve to accept an axis parameter for plotting

power_curve(trueBetas[2], FE_cca_b3SE, 100, discrep_interval, FE_cca_.df_resid, alphaLevel, "CCA", "X3", axs[0])
power_curve(trueBetas[2], FE_mi_b3SE, 100, discrep_interval, FE_mi_.df_resid, alphaLevel, "Multiple Imputation", "X3", axs[1])
power_curve(trueBetas[2], FE_lh_b3SE, 100, discrep_interval, FE_lh_.df_resid, alphaLevel, "Likelihood", "X3", axs[2])
power_curve(trueBetas[2], FE_rf_b3SE, 100, discrep_interval, FE_rf_.df_resid, alphaLevel, "Random Forest", "X3", axs[3])
###
power_curve(trueBetas[2], FE_lgb_b3SE, 100, discrep_interval, FE_lgb_.df_resid, alphaLevel, "LGBM", "X3", axs[4])
power_curve(trueBetas[2], FE_mlp_b3SE, 100, discrep_interval, FE_mlp_.df_resid, alphaLevel, "MLP", "X3", axs[5])
power_curve(trueBetas[2], FE_vae_b3SE, 100, discrep_interval, FE_vae_.df_resid, alphaLevel, "VAE", "X3", axs[6])
power_curve(trueBetas[2], FE_gae_b3SE, 100, discrep_interval, FE_gae_.df_resid, alphaLevel, "GAE", "X3", axs[7])
power_curve(trueBetas[2], FE_dif_b3SE, 100, discrep_interval, FE_dif_.df_resid, alphaLevel, "Diffusion", "X3", axs[8])

# Save the figure to a file
#plt.savefig(filepath+"Figures/Case"+str(case)+"/powerX3_FEs.png")
plt.savefig(filepath+"/powerX3_FEs.png")
plt.close(fig)

###### REl ######

# Set up the figure with 3 rows and 4 columns of subplots
fig, axs = plt.subplots(3, 3, figsize=(12, 16))
fig.suptitle('Power of X3 (Study Random Effects)', fontsize=16, fontweight='bold')
axs = axs.flatten()
#plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.1, hspace=0.4, wspace=0.4)

# Assuming power_curve is a function that plots on a given axis
# You need to define power_curve to accept an axis parameter for plotting

power_curve(trueBetas[2], RE_cca_b3SE, 100, discrep_interval, RE_cca_.df_resid, alphaLevel, "CCA", "X3", axs[0])
power_curve(trueBetas[2], RE_mi_b3SE, 100, discrep_interval, RE_mi_.df_resid, alphaLevel, "Multiple Imputation", "X3", axs[1])
power_curve(trueBetas[2], RE_lh_b3SE, 100, discrep_interval, RE_lh_.df_resid, alphaLevel, "Likelihood", "X3", axs[2])
power_curve(trueBetas[2], RE_rf_b3SE, 100, discrep_interval, RE_rf_.df_resid, alphaLevel, "Random Forest", "X3", axs[3])
###
power_curve(trueBetas[2], RE_lgb_b3SE, 100, discrep_interval, RE_lgb_.df_resid, alphaLevel, "LGBM", "X3", axs[4])
power_curve(trueBetas[2], RE_mlp_b3SE, 100, discrep_interval, RE_mlp_.df_resid, alphaLevel, "MLP", "X3", axs[5])
power_curve(trueBetas[2], RE_vae_b3SE, 100, discrep_interval, RE_vae_.df_resid, alphaLevel, "VAE", "X3", axs[6])
power_curve(trueBetas[2], RE_gae_b3SE, 100, discrep_interval, RE_gae_.df_resid, alphaLevel, "GAE", "X3", axs[7])
power_curve(trueBetas[2], RE_dif_b3SE, 100, discrep_interval, RE_dif_.df_resid, alphaLevel, "Diffusion", "X3", axs[8])

# Save the figure to a file
#plt.savefig(filepath+"Figures/Case"+str(case)+"/powerX3_RE.png")
plt.savefig(filepath+"/powerX3_RE.png")
plt.close(fig)

###### FEl ######

# Set up the figure with 3 rows and 4 columns of subplots
fig, axs = plt.subplots(3, 3, figsize=(12, 16))
fig.suptitle('Power of X3 (Time Fixed Effects)', fontsize=16, fontweight='bold')
axs = axs.flatten()
#plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.1, hspace=0.4, wspace=0.4)

# Assuming power_curve is a function that plots on a given axis
# You need to define power_curve to accept an axis parameter for plotting

power_curve(trueBetas[2], FEt_cca_b3SE, 100, discrep_interval, FEt_cca_.df_resid, alphaLevel, "CCA", "X3", axs[0])
power_curve(trueBetas[2], FEt_mi_b3SE, 100, discrep_interval, FEt_mi_.df_resid, alphaLevel, "Multiple Imputation", "X3", axs[1])
power_curve(trueBetas[2], FEt_lh_b3SE, 100, discrep_interval, FEt_lh_.df_resid, alphaLevel, "Likelihood", "X3", axs[2])
power_curve(trueBetas[2], FEt_rf_b3SE, 100, discrep_interval, FEt_rf_.df_resid, alphaLevel, "Random Forest", "X3", axs[3])
###
power_curve(trueBetas[2], FEt_lgb_b3SE, 100, discrep_interval, FEt_lgb_.df_resid, alphaLevel, "LGBM", "X3", axs[4])
power_curve(trueBetas[2], FEt_mlp_b3SE, 100, discrep_interval, FEt_mlp_.df_resid, alphaLevel, "MLP", "X3", axs[5])
power_curve(trueBetas[2], FEt_vae_b3SE, 100, discrep_interval, FEt_vae_.df_resid, alphaLevel, "VAE", "X3", axs[6])
power_curve(trueBetas[2], FEt_gae_b3SE, 100, discrep_interval, FEt_gae_.df_resid, alphaLevel, "GAE", "X3", axs[7])
power_curve(trueBetas[2], FEt_dif_b3SE, 100, discrep_interval, FEt_dif_.df_resid, alphaLevel, "Diffusion", "X3", axs[8])

# Save the figure to a file
#plt.savefig(filepath+"Figures/Case"+str(case)+"/powerX3_FEt.png")
plt.savefig(filepath+"/powerX3_FEt.png")
plt.close(fig)



#%%###########################################################################
###                                 EXPORT DATA                            ###
##############################################################################

def as_1d(x):
    return np.asarray(x).reshape(-1)

X_cca.to_csv(filepath+"/cca_X.csv", index=False)
X_mi.to_csv(filepath+"/mi_X.csv", index=False)
X_lh.to_csv(filepath+"/lh_X.csv", index=False)
X_rf.to_csv(filepath+"/rf_X.csv", index=False)
X_lgb.to_csv(filepath+"/lgb_X.csv", index=False)
X_mlp.to_csv(filepath+"/mlp_X.csv", index=False)
X_vae.to_csv(filepath+"/vae_X.csv", index=False)
X_gae.to_csv(filepath+"/gae_X.csv", index=False)
X_dif.to_csv(filepath+"/dif_X.csv", index=False)

#X_cca.to_csv(filepath+"Data/Case"+str(case)+"/cca_X.csv", index=False)
#X_mi.to_csv(filepath+"Data/Case"+str(case)+"/mi_X.csv", index=False)
#X_lh.to_csv(filepath+"Data/Case"+str(case)+"/lh_X.csv", index=False)
#X_rf.to_csv(filepath+"Data/Case"+str(case)+"/rf_X.csv", index=False)
#X_lgb.to_csv(filepath+"Data/Case"+str(case)+"/lgb_X.csv", index=False)
#X_vae.to_csv(filepath+"Data/Case"+str(case)+"/vae_X.csv", index=False)
#X_gae.to_csv(filepath+"Data/Case"+str(case)+"/gae_X.csv", index=False)
#X_dif.to_csv(filepath+"Data/Case"+str(case)+"/dif_X.csv", index=False)

arrays_to_export = {
    ### CCA ###
    "cca_y.csv": (
        [y, y - FE_rf_.resid.values, FE_rf_.resid.values, y - FEt_rf_.resid.values, FEt_rf_.resid.values, y - RE_rf_.resids.values, RE_rf_.resids.values],
        ["y", "y_minus_FE_resids", "FE_resids", "y_minus_FEt_resids", "FEt_resids", "y_minus_RE_resids", "RE_resids"]
    ),
    "cca_mse.csv": (
        [FE_cca_mse, FE_cca_mae, FE_cca_mpe, FE_cca_mape, FEt_cca_mse, FEt_cca_mae, FEt_cca_mpe, FEt_cca_mape, RE_cca_mse, RE_cca_mae, RE_cca_mpe, RE_cca_mape],
        ["FE_mse", "FE_mae", "FE_mpe", "FE_mape", "FEt_mse", "FEt_mae", "FEt_mpe", "FEt_mape", "RE_mse", "RE_mae", "RE_mpe", "RE_mape"]
    ),
    "cca_mse_x1.csv": (
        [FE_cca_mse_x1, FE_cca_mae_x1, FE_cca_mpe_x1, FE_cca_mape_x1, FEt_cca_mse_x1, FEt_cca_mae_x1, FEt_cca_mpe_x1, FEt_cca_mape_x1, RE_cca_mse_x1, RE_cca_mae_x1, RE_cca_mpe_x1, RE_cca_mape_x1],
        ["FE_mse_x1", "FE_mae_x1", "FE_mpe_x1", "FE_mape_x1", "FEt_mse_x1", "FEt_mae_x1", "FEt_mpe_x1", "FEt_mape_x1", "RE_mse_x1", "RE_mae_x1", "RE_mpe_x1", "RE_mape_x1"]
    ),
    "cca_betas.csv": (
        [FE_cca_b0, FE_cca_b1, FE_cca_b2, FE_cca_b3, FEt_cca_b0, FEt_cca_b1, FEt_cca_b2, FEt_cca_b3, RE_cca_b0, RE_cca_b1, RE_cca_b2, RE_cca_b3],
        ["FE_b0", "FE_b1", "FE_b2", "FE_b3", "FEt_b0", "FEt_b1", "FEt_b2", "FEt_b3", "RE_b0", "RE_b1", "RE_b2", "RE_b3"]
    ),
    "cca_SEs.csv": (
        [FE_cca_b0SE, FE_cca_b1SE, FE_cca_b2SE, FE_cca_b3SE, FEt_cca_b0SE, FEt_cca_b1SE, FEt_cca_b2SE, FEt_cca_b3SE, RE_cca_b0SE, RE_cca_b1SE, RE_cca_b2SE, RE_cca_b3SE],
        ["FE_b0_SE", "FE_b1_SE", "FE_b2_SE", "FE_b3_SE", "FEt_b0_SE", "FEt_b1_SE", "FEt_b2_SE", "FEt_b3_SE", "RE_b0_SE", "RE_b1_SE", "RE_b2_SE", "RE_b3_SE"]
    ),
    "cca_Vars.csv": (
        [FE_cca_b0Var, FE_cca_b1Var, FE_cca_b2Var, FE_cca_b3Var, FEt_cca_b0Var, FEt_cca_b1Var, FEt_cca_b2Var, FEt_cca_b3Var, RE_cca_b0Var, RE_cca_b1Var, RE_cca_b2Var, RE_cca_b3Var],
        ["FE_b0_Var", "FE_b1_Var", "FE_b2_Var", "FE_b3_Var", "FEt_b0_Var", "FEt_b1_Var", "FEt_b2_Var", "FEt_b3_Var", "RE_b0_Var", "RE_b1_Var", "RE_b2_Var", "RE_b3_Var"]
    ),
    "cca_aic.csv": (
        [FE_cca_aic, FEt_cca_aic, RE_cca_aic],
        ["FE_aic", "FEt_aic", "RE_aic"]
    ),
    "cca_CIs.csv": (
        [CIlo_FE_cca_b0, CIhi_FE_cca_b0, CIlo_FE_cca_b1, CIhi_FE_cca_b1, CIlo_FE_cca_b2, CIhi_FE_cca_b2, CIlo_FE_cca_b3, CIhi_FE_cca_b3,
            CIlo_FEt_cca_b0, CIhi_FEt_cca_b0, CIlo_FEt_cca_b1, CIhi_FEt_cca_b1, CIlo_FEt_cca_b2, CIhi_FEt_cca_b2, CIlo_FEt_cca_b3, CIhi_FEt_cca_b3,
            CIlo_RE_cca_b0, CIhi_RE_cca_b0, CIlo_RE_cca_b1, CIhi_RE_cca_b1, CIlo_RE_cca_b2, CIhi_RE_cca_b2, CIlo_RE_cca_b3, CIhi_RE_cca_b3],
        ["FE_b0_CI_lo", "FE_b0_CI_hi","FE_b1_CI_lo", "FE_b1_CI_hi","FE_b2_CI_lo", "FE_b2_CI_hi","FE_b3_CI_lo", "FE_b3_CI_hi",
            "FEt_b0_CI_lo", "FEt_b0_CI_hi","FEt_b1_CI_lo", "FEt_b1_CI_hi","FEt_b2_CI_lo", "FEt_b2_CI_hi","FEt_b3_CI_lo", "FEt_b3_CI_hi",
            "RE_b0_CI_lo", "RE_b0_CI_hi","RE_b1_CI_lo", "RE_b1_CI_hi","RE_b2_CI_lo", "RE_b2_CI_hi","RE_b3_CI_lo", "RE_b3_CI_hi",]
    ),
    ### MI ###
    "mi_y.csv": (
        [y, y - FE_rf_.resid.values, FE_rf_.resid.values, y - FEt_rf_.resid.values, FEt_rf_.resid.values, y - RE_rf_.resids.values, RE_rf_.resids.values],
        ["y", "y_minus_FE_resids", "FE_resids", "y_minus_FEt_resids", "FEt_resids", "y_minus_RE_resids", "RE_resids"]
    ),
    "mi_mse.csv": (
        [FE_mi_mse, FE_mi_mae, FE_mi_mpe, FE_mi_mape, FEt_mi_mse, FEt_mi_mae, FEt_mi_mpe, FEt_mi_mape, RE_mi_mse, RE_mi_mae, RE_mi_mpe, RE_mi_mape],
        ["FE_mse", "FE_mae", "FE_mpe", "FE_mape", "FEt_mse", "FEt_mae", "FEt_mpe", "FEt_mape", "RE_mse", "RE_mae", "RE_mpe", "RE_mape"]
    ),
    "mi_mse_x1.csv": (
        [FE_mi_mse_x1, FE_mi_mae_x1, FE_mi_mpe_x1, FE_mi_mape_x1, FEt_mi_mse_x1, FEt_mi_mae_x1, FEt_mi_mpe_x1, FEt_mi_mape_x1, RE_mi_mse_x1, RE_mi_mae_x1, RE_mi_mpe_x1, RE_mi_mape_x1],
        ["FE_mse_x1", "FE_mae_x1", "FE_mpe_x1", "FE_mape_x1", "FEt_mse_x1", "FEt_mae_x1", "FEt_mpe_x1", "FEt_mape_x1", "RE_mse_x1", "RE_mae_x1", "RE_mpe_x1", "RE_mape_x1"]
    ),
    "mi_betas.csv": (
        [FE_mi_b0, FE_mi_b1, FE_mi_b2, FE_mi_b3, FEt_mi_b0, FEt_mi_b1, FEt_mi_b2, FEt_mi_b3, RE_mi_b0, RE_mi_b1, RE_mi_b2, RE_mi_b3],
        ["FE_b0", "FE_b1", "FE_b2", "FE_b3", "FEt_b0", "FEt_b1", "FEt_b2", "FEt_b3", "RE_b0", "RE_b1", "RE_b2", "RE_b3"]
    ),
    "mi_SEs.csv": (
        [FE_mi_b0SE, FE_mi_b1SE, FE_mi_b2SE, FE_mi_b3SE, FEt_mi_b0SE, FEt_mi_b1SE, FEt_mi_b2SE, FEt_mi_b3SE, RE_mi_b0SE, RE_mi_b1SE, RE_mi_b2SE, RE_mi_b3SE],
        ["FE_b0_SE", "FE_b1_SE", "FE_b2_SE", "FE_b3_SE", "FEt_b0_SE", "FEt_b1_SE", "FEt_b2_SE", "FEt_b3_SE", "RE_b0_SE", "RE_b1_SE", "RE_b2_SE", "RE_b3_SE"]
    ),
    "mi_Vars.csv": (
        [FE_mi_b0Var, FE_mi_b1Var, FE_mi_b2Var, FE_mi_b3Var, FEt_mi_b0Var, FEt_mi_b1Var, FEt_mi_b2Var, FEt_mi_b3Var, RE_mi_b0Var, RE_mi_b1Var, RE_mi_b2Var, RE_mi_b3Var],
        ["FE_b0_Var", "FE_b1_Var", "FE_b2_Var", "FE_b3_Var", "FEt_b0_Var", "FEt_b1_Var", "FEt_b2_Var", "FEt_b3_Var", "RE_b0_Var", "RE_b1_Var", "RE_b2_Var", "RE_b3_Var"]
    ),
    "mi_aic.csv": (
        [FE_mi_aic, FEt_mi_aic, RE_mi_aic],
        ["FE_aic", "FEt_aic", "RE_aic"]
    ),
    "mi_CIs.csv": (
        [CIlo_FE_mi_b0, CIhi_FE_mi_b0, CIlo_FE_mi_b1, CIhi_FE_mi_b1, CIlo_FE_mi_b2, CIhi_FE_mi_b2, CIlo_FE_mi_b3, CIhi_FE_mi_b3,
            CIlo_FEt_mi_b0, CIhi_FEt_mi_b0, CIlo_FEt_mi_b1, CIhi_FEt_mi_b1, CIlo_FEt_mi_b2, CIhi_FEt_mi_b2, CIlo_FEt_mi_b3, CIhi_FEt_mi_b3,
            CIlo_RE_mi_b0, CIhi_RE_mi_b0, CIlo_RE_mi_b1, CIhi_RE_mi_b1, CIlo_RE_mi_b2, CIhi_RE_mi_b2, CIlo_RE_mi_b3, CIhi_RE_mi_b3],
        ["FE_b0_CI_lo", "FE_b0_CI_hi","FE_b1_CI_lo", "FE_b1_CI_hi","FE_b2_CI_lo", "FE_b2_CI_hi","FE_b3_CI_lo", "FE_b3_CI_hi",
            "FEt_b0_CI_lo", "FEt_b0_CI_hi","FEt_b1_CI_lo", "FEt_b1_CI_hi","FEt_b2_CI_lo", "FEt_b2_CI_hi","FEt_b3_CI_lo", "FEt_b3_CI_hi",
            "RE_b0_CI_lo", "RE_b0_CI_hi","RE_b1_CI_lo", "RE_b1_CI_hi","RE_b2_CI_lo", "RE_b2_CI_hi","RE_b3_CI_lo", "RE_b3_CI_hi",]
    ),
    ### LH ###
    "lh_y.csv": (
        [y, y - FE_rf_.resid.values, FE_rf_.resid.values, y - FEt_rf_.resid.values, FEt_rf_.resid.values, y - RE_rf_.resids.values, RE_rf_.resids.values],
        ["y", "y_minus_FE_resids", "FE_resids", "y_minus_FEt_resids", "FEt_resids", "y_minus_RE_resids", "RE_resids"]
    ),
    "lh_mse.csv": (
        [FE_lh_mse, FE_lh_mae, FE_lh_mpe, FE_lh_mape, FEt_lh_mse, FEt_lh_mae, FEt_lh_mpe, FEt_lh_mape, RE_lh_mse, RE_lh_mae, RE_lh_mpe, RE_lh_mape],
        ["FE_mse", "FE_mae", "FE_mpe", "FE_mape", "FEt_mse", "FEt_mae", "FEt_mpe", "FEt_mape", "RE_mse", "RE_mae", "RE_mpe", "RE_mape"]
    ),
    "lh_mse_x1.csv": (
        [FE_lh_mse_x1, FE_lh_mae_x1, FE_lh_mpe_x1, FE_lh_mape_x1, FEt_lh_mse_x1, FEt_lh_mae_x1, FEt_lh_mpe_x1, FEt_lh_mape_x1, RE_lh_mse_x1, RE_lh_mae_x1, RE_lh_mpe_x1, RE_lh_mape_x1],
        ["FE_mse_x1", "FE_mae_x1", "FE_mpe_x1", "FE_mape_x1", "FEt_mse_x1", "FEt_mae_x1", "FEt_mpe_x1", "FEt_mape_x1", "RE_mse_x1", "RE_mae_x1", "RE_mpe_x1", "RE_mape_x1"]
    ),
    "lh_betas.csv": (
        [FE_lh_b0, FE_lh_b1, FE_lh_b2, FE_lh_b3, FEt_lh_b0, FEt_lh_b1, FEt_lh_b2, FEt_lh_b3, RE_lh_b0, RE_lh_b1, RE_lh_b2, RE_lh_b3],
        ["FE_b0", "FE_b1", "FE_b2", "FE_b3", "FEt_b0", "FEt_b1", "FEt_b2", "FEt_b3", "RE_b0", "RE_b1", "RE_b2", "RE_b3"]
    ),
    "lh_SEs.csv": (
        [FE_lh_b0SE, FE_lh_b1SE, FE_lh_b2SE, FE_lh_b3SE, FEt_lh_b0SE, FEt_lh_b1SE, FEt_lh_b2SE, FEt_lh_b3SE, RE_lh_b0SE, RE_lh_b1SE, RE_lh_b2SE, RE_lh_b3SE],
        ["FE_b0_SE", "FE_b1_SE", "FE_b2_SE", "FE_b3_SE", "FEt_b0_SE", "FEt_b1_SE", "FEt_b2_SE", "FEt_b3_SE", "RE_b0_SE", "RE_b1_SE", "RE_b2_SE", "RE_b3_SE"]
    ),
    "lh_Vars.csv": (
        [FE_lh_b0Var, FE_lh_b1Var, FE_lh_b2Var, FE_lh_b3Var, FEt_lh_b0Var, FEt_lh_b1Var, FEt_lh_b2Var, FEt_lh_b3Var, RE_lh_b0Var, RE_lh_b1Var, RE_lh_b2Var, RE_lh_b3Var],
        ["FE_b0_Var", "FE_b1_Var", "FE_b2_Var", "FE_b3_Var", "FEt_b0_Var", "FEt_b1_Var", "FEt_b2_Var", "FEt_b3_Var", "RE_b0_Var", "RE_b1_Var", "RE_b2_Var", "RE_b3_Var"]
    ),
    "lh_aic.csv": (
        [FE_lh_aic, FEt_lh_aic, RE_lh_aic],
        ["FE_aic", "FEt_aic", "RE_aic"]
    ),
    "lh_CIs.csv": (
        [CIlo_FE_lh_b0, CIhi_FE_lh_b0, CIlo_FE_lh_b1, CIhi_FE_lh_b1, CIlo_FE_lh_b2, CIhi_FE_lh_b2, CIlo_FE_lh_b3, CIhi_FE_lh_b3,
            CIlo_FEt_lh_b0, CIhi_FEt_lh_b0, CIlo_FEt_lh_b1, CIhi_FEt_lh_b1, CIlo_FEt_lh_b2, CIhi_FEt_lh_b2, CIlo_FEt_lh_b3, CIhi_FEt_lh_b3,
            CIlo_RE_lh_b0, CIhi_RE_lh_b0, CIlo_RE_lh_b1, CIhi_RE_lh_b1, CIlo_RE_lh_b2, CIhi_RE_lh_b2, CIlo_RE_lh_b3, CIhi_RE_lh_b3],
        ["FE_b0_CI_lo", "FE_b0_CI_hi","FE_b1_CI_lo", "FE_b1_CI_hi","FE_b2_CI_lo", "FE_b2_CI_hi","FE_b3_CI_lo", "FE_b3_CI_hi",
            "FEt_b0_CI_lo", "FEt_b0_CI_hi","FEt_b1_CI_lo", "FEt_b1_CI_hi","FEt_b2_CI_lo", "FEt_b2_CI_hi","FEt_b3_CI_lo", "FEt_b3_CI_hi",
            "RE_b0_CI_lo", "RE_b0_CI_hi","RE_b1_CI_lo", "RE_b1_CI_hi","RE_b2_CI_lo", "RE_b2_CI_hi","RE_b3_CI_lo", "RE_b3_CI_hi",]
    ),
    ### RF ###
    "rf_y.csv": (
        [y, y - FE_rf_.resid.values, FE_rf_.resid.values, y - FEt_rf_.resid.values, FEt_rf_.resid.values, y - RE_rf_.resids.values, RE_rf_.resids.values],
        ["y", "y_minus_FE_resids", "FE_resids", "y_minus_FEt_resids", "FEt_resids", "y_minus_RE_resids", "RE_resids"]
    ),
    "rf_mse.csv": (
        [FE_rf_mse, FE_rf_mae, FE_rf_mpe, FE_rf_mape, FEt_rf_mse, FEt_rf_mae, FEt_rf_mpe, FEt_rf_mape, RE_rf_mse, RE_rf_mae, RE_rf_mpe, RE_rf_mape],
        ["FE_mse", "FE_mae", "FE_mpe", "FE_mape", "FEt_mse", "FEt_mae", "FEt_mpe", "FEt_mape", "RE_mse", "RE_mae", "RE_mpe", "RE_mape"]
    ),
    "rf_mse_x1.csv": (
        [FE_rf_mse_x1, FE_rf_mae_x1, FE_rf_mpe_x1, FE_rf_mape_x1, FEt_rf_mse_x1, FEt_rf_mae_x1, FEt_rf_mpe_x1, FEt_rf_mape_x1, RE_rf_mse_x1, RE_rf_mae_x1, RE_rf_mpe_x1, RE_rf_mape_x1],
        ["FE_mse_x1", "FE_mae_x1", "FE_mpe_x1", "FE_mape_x1", "FEt_mse_x1", "FEt_mae_x1", "FEt_mpe_x1", "FEt_mape_x1", "RE_mse_x1", "RE_mae_x1", "RE_mpe_x1", "RE_mape_x1"]
    ),
    "rf_betas.csv": (
        [FE_rf_b0, FE_rf_b1, FE_rf_b2, FE_rf_b3, FEt_rf_b0, FEt_rf_b1, FEt_rf_b2, FEt_rf_b3, RE_rf_b0, RE_rf_b1, RE_rf_b2, RE_rf_b3],
        ["FE_b0", "FE_b1", "FE_b2", "FE_b3", "FEt_b0", "FEt_b1", "FEt_b2", "FEt_b3", "RE_b0", "RE_b1", "RE_b2", "RE_b3"]
    ),
    "rf_SEs.csv": (
        [FE_rf_b0SE, FE_rf_b1SE, FE_rf_b2SE, FE_rf_b3SE, FEt_rf_b0SE, FEt_rf_b1SE, FEt_rf_b2SE, FEt_rf_b3SE, RE_rf_b0SE, RE_rf_b1SE, RE_rf_b2SE, RE_rf_b3SE],
        ["FE_b0_SE", "FE_b1_SE", "FE_b2_SE", "FE_b3_SE", "FEt_b0_SE", "FEt_b1_SE", "FEt_b2_SE", "FEt_b3_SE", "RE_b0_SE", "RE_b1_SE", "RE_b2_SE", "RE_b3_SE"]
    ),
    "rf_Vars.csv": (
        [FE_rf_b0Var, FE_rf_b1Var, FE_rf_b2Var, FE_rf_b3Var, FEt_rf_b0Var, FEt_rf_b1Var, FEt_rf_b2Var, FEt_rf_b3Var, RE_rf_b0Var, RE_rf_b1Var, RE_rf_b2Var, RE_rf_b3Var],
        ["FE_b0_Var", "FE_b1_Var", "FE_b2_Var", "FE_b3_Var", "FEt_b0_Var", "FEt_b1_Var", "FEt_b2_Var", "FEt_b3_Var", "RE_b0_Var", "RE_b1_Var", "RE_b2_Var", "RE_b3_Var"]
    ),
    "rf_aic.csv": (
        [FE_rf_aic, FEt_rf_aic, RE_rf_aic],
        ["FE_aic", "FEt_aic", "RE_aic"]
    ),
    "rf_CIs.csv": (
        [CIlo_FE_rf_b0, CIhi_FE_rf_b0, CIlo_FE_rf_b1, CIhi_FE_rf_b1, CIlo_FE_rf_b2, CIhi_FE_rf_b2, CIlo_FE_rf_b3, CIhi_FE_rf_b3,
            CIlo_FEt_rf_b0, CIhi_FEt_rf_b0, CIlo_FEt_rf_b1, CIhi_FEt_rf_b1, CIlo_FEt_rf_b2, CIhi_FEt_rf_b2, CIlo_FEt_rf_b3, CIhi_FEt_rf_b3,
            CIlo_RE_rf_b0, CIhi_RE_rf_b0, CIlo_RE_rf_b1, CIhi_RE_rf_b1, CIlo_RE_rf_b2, CIhi_RE_rf_b2, CIlo_RE_rf_b3, CIhi_RE_rf_b3],
        ["FE_b0_CI_lo", "FE_b0_CI_hi","FE_b1_CI_lo", "FE_b1_CI_hi","FE_b2_CI_lo", "FE_b2_CI_hi","FE_b3_CI_lo", "FE_b3_CI_hi",
            "FEt_b0_CI_lo", "FEt_b0_CI_hi","FEt_b1_CI_lo", "FEt_b1_CI_hi","FEt_b2_CI_lo", "FEt_b2_CI_hi","FEt_b3_CI_lo", "FEt_b3_CI_hi",
            "RE_b0_CI_lo", "RE_b0_CI_hi","RE_b1_CI_lo", "RE_b1_CI_hi","RE_b2_CI_lo", "RE_b2_CI_hi","RE_b3_CI_lo", "RE_b3_CI_hi",]
    ),
    ### LGB ###
    "lgb_y.csv": (
        [y, y - FE_rf_.resid.values, FE_rf_.resid.values, y - FEt_rf_.resid.values, FEt_rf_.resid.values, y - RE_rf_.resids.values, RE_rf_.resids.values],
        ["y", "y_minus_FE_resids", "FE_resids", "y_minus_FEt_resids", "FEt_resids", "y_minus_RE_resids", "RE_resids"]
    ),
    "lgb_mse.csv": (
        [FE_lgb_mse, FE_lgb_mae, FE_lgb_mpe, FE_lgb_mape, FEt_lgb_mse, FEt_lgb_mae, FEt_lgb_mpe, FEt_lgb_mape, RE_lgb_mse, RE_lgb_mae, RE_lgb_mpe, RE_lgb_mape],
        ["FE_mse", "FE_mae", "FE_mpe", "FE_mape", "FEt_mse", "FEt_mae", "FEt_mpe", "FEt_mape", "RE_mse", "RE_mae", "RE_mpe", "RE_mape"]
    ),
    "lgb_mse_x1.csv": (
        [FE_lgb_mse_x1, FE_lgb_mae_x1, FE_lgb_mpe_x1, FE_lgb_mape_x1, FEt_lgb_mse_x1, FEt_lgb_mae_x1, FEt_lgb_mpe_x1, FEt_lgb_mape_x1, RE_lgb_mse_x1, RE_lgb_mae_x1, RE_lgb_mpe_x1, RE_lgb_mape_x1],
        ["FE_mse_x1", "FE_mae_x1", "FE_mpe_x1", "FE_mape_x1", "FEt_mse_x1", "FEt_mae_x1", "FEt_mpe_x1", "FEt_mape_x1", "RE_mse_x1", "RE_mae_x1", "RE_mpe_x1", "RE_mape_x1"]
    ),
    "lgb_betas.csv": (
        [FE_lgb_b0, FE_lgb_b1, FE_lgb_b2, FE_lgb_b3, FEt_lgb_b0, FEt_lgb_b1, FEt_lgb_b2, FEt_lgb_b3, RE_lgb_b0, RE_lgb_b1, RE_lgb_b2, RE_lgb_b3],
        ["FE_b0", "FE_b1", "FE_b2", "FE_b3", "FEt_b0", "FEt_b1", "FEt_b2", "FEt_b3", "RE_b0", "RE_b1", "RE_b2", "RE_b3"]
    ),
    "lgb_SEs.csv": (
        [FE_lgb_b0SE, FE_lgb_b1SE, FE_lgb_b2SE, FE_lgb_b3SE, FEt_lgb_b0SE, FEt_lgb_b1SE, FEt_lgb_b2SE, FEt_lgb_b3SE, RE_lgb_b0SE, RE_lgb_b1SE, RE_lgb_b2SE, RE_lgb_b3SE],
        ["FE_b0_SE", "FE_b1_SE", "FE_b2_SE", "FE_b3_SE", "FEt_b0_SE", "FEt_b1_SE", "FEt_b2_SE", "FEt_b3_SE", "RE_b0_SE", "RE_b1_SE", "RE_b2_SE", "RE_b3_SE"]
    ),
    "lgb_Vars.csv": (
        [FE_lgb_b0Var, FE_lgb_b1Var, FE_lgb_b2Var, FE_lgb_b3Var, FEt_lgb_b0Var, FEt_lgb_b1Var, FEt_lgb_b2Var, FEt_lgb_b3Var, RE_lgb_b0Var, RE_lgb_b1Var, RE_lgb_b2Var, RE_lgb_b3Var],
        ["FE_b0_Var", "FE_b1_Var", "FE_b2_Var", "FE_b3_Var", "FEt_b0_Var", "FEt_b1_Var", "FEt_b2_Var", "FEt_b3_Var", "RE_b0_Var", "RE_b1_Var", "RE_b2_Var", "RE_b3_Var"]
    ),
    "lgb_aic.csv": (
        [FE_lgb_aic, FEt_lgb_aic, RE_lgb_aic],
        ["FE_aic", "FEt_aic", "RE_aic"]
    ),
    "lgb_CIs.csv": (
        [CIlo_FE_lgb_b0, CIhi_FE_lgb_b0, CIlo_FE_lgb_b1, CIhi_FE_lgb_b1, CIlo_FE_lgb_b2, CIhi_FE_lgb_b2, CIlo_FE_lgb_b3, CIhi_FE_lgb_b3,
            CIlo_FEt_lgb_b0, CIhi_FEt_lgb_b0, CIlo_FEt_lgb_b1, CIhi_FEt_lgb_b1, CIlo_FEt_lgb_b2, CIhi_FEt_lgb_b2, CIlo_FEt_lgb_b3, CIhi_FEt_lgb_b3,
            CIlo_RE_lgb_b0, CIhi_RE_lgb_b0, CIlo_RE_lgb_b1, CIhi_RE_lgb_b1, CIlo_RE_lgb_b2, CIhi_RE_lgb_b2, CIlo_RE_lgb_b3, CIhi_RE_lgb_b3],
        ["FE_b0_CI_lo", "FE_b0_CI_hi","FE_b1_CI_lo", "FE_b1_CI_hi","FE_b2_CI_lo", "FE_b2_CI_hi","FE_b3_CI_lo", "FE_b3_CI_hi",
            "FEt_b0_CI_lo", "FEt_b0_CI_hi","FEt_b1_CI_lo", "FEt_b1_CI_hi","FEt_b2_CI_lo", "FEt_b2_CI_hi","FEt_b3_CI_lo", "FEt_b3_CI_hi",
            "RE_b0_CI_lo", "RE_b0_CI_hi","RE_b1_CI_lo", "RE_b1_CI_hi","RE_b2_CI_lo", "RE_b2_CI_hi","RE_b3_CI_lo", "RE_b3_CI_hi",]
    ),
    ### MLP ###
    "mlp_y.csv": (
        [y, y - FE_rf_.resid.values, FE_rf_.resid.values, y - FEt_rf_.resid.values, FEt_rf_.resid.values, y - RE_rf_.resids.values, RE_rf_.resids.values],
        ["y", "y_minus_FE_resids", "FE_resids", "y_minus_FEt_resids", "FEt_resids", "y_minus_RE_resids", "RE_resids"]
    ),
    "mlp_mse.csv": (
        [FE_mlp_mse, FE_mlp_mae, FE_mlp_mpe, FE_mlp_mape, FEt_mlp_mse, FEt_mlp_mae, FEt_mlp_mpe, FEt_mlp_mape, RE_mlp_mse, RE_mlp_mae, RE_mlp_mpe, RE_mlp_mape],
        ["FE_mse", "FE_mae", "FE_mpe", "FE_mape", "FEt_mse", "FEt_mae", "FEt_mpe", "FEt_mape", "RE_mse", "RE_mae", "RE_mpe", "RE_mape"]
    ),
    "mlp_mse_x1.csv": (
        [FE_mlp_mse_x1, FE_mlp_mae_x1, FE_mlp_mpe_x1, FE_mlp_mape_x1, FEt_mlp_mse_x1, FEt_mlp_mae_x1, FEt_mlp_mpe_x1, FEt_mlp_mape_x1, RE_mlp_mse_x1, RE_mlp_mae_x1, RE_mlp_mpe_x1, RE_mlp_mape_x1],
        ["FE_mse_x1", "FE_mae_x1", "FE_mpe_x1", "FE_mape_x1", "FEt_mse_x1", "FEt_mae_x1", "FEt_mpe_x1", "FEt_mape_x1", "RE_mse_x1", "RE_mae_x1", "RE_mpe_x1", "RE_mape_x1"]
    ),
    "mlp_betas.csv": (
        [FE_mlp_b0, FE_mlp_b1, FE_mlp_b2, FE_mlp_b3, FEt_mlp_b0, FEt_mlp_b1, FEt_mlp_b2, FEt_mlp_b3, RE_mlp_b0, RE_mlp_b1, RE_mlp_b2, RE_mlp_b3],
        ["FE_b0", "FE_b1", "FE_b2", "FE_b3", "FEt_b0", "FEt_b1", "FEt_b2", "FEt_b3", "RE_b0", "RE_b1", "RE_b2", "RE_b3"]
    ),
    "mlp_SEs.csv": (
        [FE_mlp_b0SE, FE_mlp_b1SE, FE_mlp_b2SE, FE_mlp_b3SE, FEt_mlp_b0SE, FEt_mlp_b1SE, FEt_mlp_b2SE, FEt_mlp_b3SE, RE_mlp_b0SE, RE_mlp_b1SE, RE_mlp_b2SE, RE_mlp_b3SE],
        ["FE_b0_SE", "FE_b1_SE", "FE_b2_SE", "FE_b3_SE", "FEt_b0_SE", "FEt_b1_SE", "FEt_b2_SE", "FEt_b3_SE", "RE_b0_SE", "RE_b1_SE", "RE_b2_SE", "RE_b3_SE"]
    ),
    "mlp_Vars.csv": (
        [FE_mlp_b0Var, FE_mlp_b1Var, FE_mlp_b2Var, FE_mlp_b3Var, FEt_mlp_b0Var, FEt_mlp_b1Var, FEt_mlp_b2Var, FEt_mlp_b3Var, RE_mlp_b0Var, RE_mlp_b1Var, RE_mlp_b2Var, RE_mlp_b3Var],
        ["FE_b0_Var", "FE_b1_Var", "FE_b2_Var", "FE_b3_Var", "FEt_b0_Var", "FEt_b1_Var", "FEt_b2_Var", "FEt_b3_Var", "RE_b0_Var", "RE_b1_Var", "RE_b2_Var", "RE_b3_Var"]
    ),
    "mlp_aic.csv": (
        [FE_mlp_aic, FEt_mlp_aic, RE_mlp_aic],
        ["FE_aic", "FEt_aic", "RE_aic"]
    ),
    "mlp_CIs.csv": (
        [CIlo_FE_mlp_b0, CIhi_FE_mlp_b0, CIlo_FE_mlp_b1, CIhi_FE_mlp_b1, CIlo_FE_mlp_b2, CIhi_FE_mlp_b2, CIlo_FE_mlp_b3, CIhi_FE_mlp_b3,
            CIlo_FEt_mlp_b0, CIhi_FEt_mlp_b0, CIlo_FEt_mlp_b1, CIhi_FEt_mlp_b1, CIlo_FEt_mlp_b2, CIhi_FEt_mlp_b2, CIlo_FEt_mlp_b3, CIhi_FEt_mlp_b3,
            CIlo_RE_mlp_b0, CIhi_RE_mlp_b0, CIlo_RE_mlp_b1, CIhi_RE_mlp_b1, CIlo_RE_mlp_b2, CIhi_RE_mlp_b2, CIlo_RE_mlp_b3, CIhi_RE_mlp_b3],
        ["FE_b0_CI_lo", "FE_b0_CI_hi","FE_b1_CI_lo", "FE_b1_CI_hi","FE_b2_CI_lo", "FE_b2_CI_hi","FE_b3_CI_lo", "FE_b3_CI_hi",
            "FEt_b0_CI_lo", "FEt_b0_CI_hi","FEt_b1_CI_lo", "FEt_b1_CI_hi","FEt_b2_CI_lo", "FEt_b2_CI_hi","FEt_b3_CI_lo", "FEt_b3_CI_hi",
            "RE_b0_CI_lo", "RE_b0_CI_hi","RE_b1_CI_lo", "RE_b1_CI_hi","RE_b2_CI_lo", "RE_b2_CI_hi","RE_b3_CI_lo", "RE_b3_CI_hi",]
    ),
    ### VAE ###
    "vae_y.csv": (
        [y, y - FE_rf_.resid.values, FE_rf_.resid.values, y - FEt_rf_.resid.values, FEt_rf_.resid.values, y - RE_rf_.resids.values, RE_rf_.resids.values],
        ["y", "y_minus_FE_resids", "FE_resids", "y_minus_FEt_resids", "FEt_resids", "y_minus_RE_resids", "RE_resids"]
    ),
    "vae_mse.csv": (
        [FE_vae_mse, FE_vae_mae, FE_vae_mpe, FE_vae_mape, FEt_vae_mse, FEt_vae_mae, FEt_vae_mpe, FEt_vae_mape, RE_vae_mse, RE_vae_mae, RE_vae_mpe, RE_vae_mape],
        ["FE_mse", "FE_mae", "FE_mpe", "FE_mape", "FEt_mse", "FEt_mae", "FEt_mpe", "FEt_mape", "RE_mse", "RE_mae", "RE_mpe", "RE_mape"]
    ),
    "vae_mse_x1.csv": (
        [FE_vae_mse_x1, FE_vae_mae_x1, FE_vae_mpe_x1, FE_vae_mape_x1, FEt_vae_mse_x1, FEt_vae_mae_x1, FEt_vae_mpe_x1, FEt_vae_mape_x1, RE_vae_mse_x1, RE_vae_mae_x1, RE_vae_mpe_x1, RE_vae_mape_x1],
        ["FE_mse_x1", "FE_mae_x1", "FE_mpe_x1", "FE_mape_x1", "FEt_mse_x1", "FEt_mae_x1", "FEt_mpe_x1", "FEt_mape_x1", "RE_mse_x1", "RE_mae_x1", "RE_mpe_x1", "RE_mape_x1"]
    ),
    "vae_betas.csv": (
        [FE_vae_b0, FE_vae_b1, FE_vae_b2, FE_vae_b3, FEt_vae_b0, FEt_vae_b1, FEt_vae_b2, FEt_vae_b3, RE_vae_b0, RE_vae_b1, RE_vae_b2, RE_vae_b3],
        ["FE_b0", "FE_b1", "FE_b2", "FE_b3", "FEt_b0", "FEt_b1", "FEt_b2", "FEt_b3", "RE_b0", "RE_b1", "RE_b2", "RE_b3"]
    ),
    "vae_SEs.csv": (
        [FE_vae_b0SE, FE_vae_b1SE, FE_vae_b2SE, FE_vae_b3SE, FEt_vae_b0SE, FEt_vae_b1SE, FEt_vae_b2SE, FEt_vae_b3SE, RE_vae_b0SE, RE_vae_b1SE, RE_vae_b2SE, RE_vae_b3SE],
        ["FE_b0_SE", "FE_b1_SE", "FE_b2_SE", "FE_b3_SE", "FEt_b0_SE", "FEt_b1_SE", "FEt_b2_SE", "FEt_b3_SE", "RE_b0_SE", "RE_b1_SE", "RE_b2_SE", "RE_b3_SE"]
    ),
    "vae_Vars.csv": (
        [FE_vae_b0Var, FE_vae_b1Var, FE_vae_b2Var, FE_vae_b3Var, FEt_vae_b0Var, FEt_vae_b1Var, FEt_vae_b2Var, FEt_vae_b3Var, RE_vae_b0Var, RE_vae_b1Var, RE_vae_b2Var, RE_vae_b3Var],
        ["FE_b0_Var", "FE_b1_Var", "FE_b2_Var", "FE_b3_Var", "FEt_b0_Var", "FEt_b1_Var", "FEt_b2_Var", "FEt_b3_Var", "RE_b0_Var", "RE_b1_Var", "RE_b2_Var", "RE_b3_Var"]
    ),
    "vae_aic.csv": (
        [FE_vae_aic, FEt_vae_aic, RE_vae_aic],
        ["FE_aic", "FEt_aic", "RE_aic"]
    ),
    "vae_CIs.csv": (
        [CIlo_FE_vae_b0, CIhi_FE_vae_b0, CIlo_FE_vae_b1, CIhi_FE_vae_b1, CIlo_FE_vae_b2, CIhi_FE_vae_b2, CIlo_FE_vae_b3, CIhi_FE_vae_b3,
            CIlo_FEt_vae_b0, CIhi_FEt_vae_b0, CIlo_FEt_vae_b1, CIhi_FEt_vae_b1, CIlo_FEt_vae_b2, CIhi_FEt_vae_b2, CIlo_FEt_vae_b3, CIhi_FEt_vae_b3,
            CIlo_RE_vae_b0, CIhi_RE_vae_b0, CIlo_RE_vae_b1, CIhi_RE_vae_b1, CIlo_RE_vae_b2, CIhi_RE_vae_b2, CIlo_RE_vae_b3, CIhi_RE_vae_b3],
        ["FE_b0_CI_lo", "FE_b0_CI_hi","FE_b1_CI_lo", "FE_b1_CI_hi","FE_b2_CI_lo", "FE_b2_CI_hi","FE_b3_CI_lo", "FE_b3_CI_hi",
            "FEt_b0_CI_lo", "FEt_b0_CI_hi","FEt_b1_CI_lo", "FEt_b1_CI_hi","FEt_b2_CI_lo", "FEt_b2_CI_hi","FEt_b3_CI_lo", "FEt_b3_CI_hi",
            "RE_b0_CI_lo", "RE_b0_CI_hi","RE_b1_CI_lo", "RE_b1_CI_hi","RE_b2_CI_lo", "RE_b2_CI_hi","RE_b3_CI_lo", "RE_b3_CI_hi",]
    ),
    ### GAE ###
    "gae_y.csv": (
        [y, y - FE_rf_.resid.values, FE_rf_.resid.values, y - FEt_rf_.resid.values, FEt_rf_.resid.values, y - RE_rf_.resids.values, RE_rf_.resids.values],
        ["y", "y_minus_FE_resids", "FE_resids", "y_minus_FEt_resids", "FEt_resids", "y_minus_RE_resids", "RE_resids"]
    ),
    "gae_mse.csv": (
        [FE_gae_mse, FE_gae_mae, FE_gae_mpe, FE_gae_mape, FEt_gae_mse, FEt_gae_mae, FEt_gae_mpe, FEt_gae_mape, RE_gae_mse, RE_gae_mae, RE_gae_mpe, RE_gae_mape],
        ["FE_mse", "FE_mae", "FE_mpe", "FE_mape", "FEt_mse", "FEt_mae", "FEt_mpe", "FEt_mape", "RE_mse", "RE_mae", "RE_mpe", "RE_mape"]
    ),
    "gae_mse_x1.csv": (
        [FE_gae_mse_x1, FE_gae_mae_x1, FE_gae_mpe_x1, FE_gae_mape_x1, FEt_gae_mse_x1, FEt_gae_mae_x1, FEt_gae_mpe_x1, FEt_gae_mape_x1, RE_gae_mse_x1, RE_gae_mae_x1, RE_gae_mpe_x1, RE_gae_mape_x1],
        ["FE_mse_x1", "FE_mae_x1", "FE_mpe_x1", "FE_mape_x1", "FEt_mse_x1", "FEt_mae_x1", "FEt_mpe_x1", "FEt_mape_x1", "RE_mse_x1", "RE_mae_x1", "RE_mpe_x1", "RE_mape_x1"]
    ),
    "gae_betas.csv": (
        [FE_gae_b0, FE_gae_b1, FE_gae_b2, FE_gae_b3, FEt_gae_b0, FEt_gae_b1, FEt_gae_b2, FEt_gae_b3, RE_gae_b0, RE_gae_b1, RE_gae_b2, RE_gae_b3],
        ["FE_b0", "FE_b1", "FE_b2", "FE_b3", "FEt_b0", "FEt_b1", "FEt_b2", "FEt_b3", "RE_b0", "RE_b1", "RE_b2", "RE_b3"]
    ),
    "gae_SEs.csv": (
        [FE_gae_b0SE, FE_gae_b1SE, FE_gae_b2SE, FE_gae_b3SE, FEt_gae_b0SE, FEt_gae_b1SE, FEt_gae_b2SE, FEt_gae_b3SE, RE_gae_b0SE, RE_gae_b1SE, RE_gae_b2SE, RE_gae_b3SE],
        ["FE_b0_SE", "FE_b1_SE", "FE_b2_SE", "FE_b3_SE", "FEt_b0_SE", "FEt_b1_SE", "FEt_b2_SE", "FEt_b3_SE", "RE_b0_SE", "RE_b1_SE", "RE_b2_SE", "RE_b3_SE"]
    ),
    "gae_Vars.csv": (
        [FE_gae_b0Var, FE_gae_b1Var, FE_gae_b2Var, FE_gae_b3Var, FEt_gae_b0Var, FEt_gae_b1Var, FEt_gae_b2Var, FEt_gae_b3Var, RE_gae_b0Var, RE_gae_b1Var, RE_gae_b2Var, RE_gae_b3Var],
        ["FE_b0_Var", "FE_b1_Var", "FE_b2_Var", "FE_b3_Var", "FEt_b0_Var", "FEt_b1_Var", "FEt_b2_Var", "FEt_b3_Var", "RE_b0_Var", "RE_b1_Var", "RE_b2_Var", "RE_b3_Var"]
    ),
    "gae_aic.csv": (
        [FE_gae_aic, FEt_gae_aic, RE_gae_aic],
        ["FE_aic", "FEt_aic", "RE_aic"]
    ),
    "gae_CIs.csv": (
        [CIlo_FE_gae_b0, CIhi_FE_gae_b0, CIlo_FE_gae_b1, CIhi_FE_gae_b1, CIlo_FE_gae_b2, CIhi_FE_gae_b2, CIlo_FE_gae_b3, CIhi_FE_gae_b3,
            CIlo_FEt_gae_b0, CIhi_FEt_gae_b0, CIlo_FEt_gae_b1, CIhi_FEt_gae_b1, CIlo_FEt_gae_b2, CIhi_FEt_gae_b2, CIlo_FEt_gae_b3, CIhi_FEt_gae_b3,
            CIlo_RE_gae_b0, CIhi_RE_gae_b0, CIlo_RE_gae_b1, CIhi_RE_gae_b1, CIlo_RE_gae_b2, CIhi_RE_gae_b2, CIlo_RE_gae_b3, CIhi_RE_gae_b3],
        ["FE_b0_CI_lo", "FE_b0_CI_hi","FE_b1_CI_lo", "FE_b1_CI_hi","FE_b2_CI_lo", "FE_b2_CI_hi","FE_b3_CI_lo", "FE_b3_CI_hi",
            "FEt_b0_CI_lo", "FEt_b0_CI_hi","FEt_b1_CI_lo", "FEt_b1_CI_hi","FEt_b2_CI_lo", "FEt_b2_CI_hi","FEt_b3_CI_lo", "FEt_b3_CI_hi",
            "RE_b0_CI_lo", "RE_b0_CI_hi","RE_b1_CI_lo", "RE_b1_CI_hi","RE_b2_CI_lo", "RE_b2_CI_hi","RE_b3_CI_lo", "RE_b3_CI_hi",]
    ),
    ### DIF ###
    "dif_y.csv": (
        [y, y - FE_rf_.resid.values, FE_rf_.resid.values, y - FEt_rf_.resid.values, FEt_rf_.resid.values, y - RE_rf_.resids.values, RE_rf_.resids.values],
        ["y", "y_minus_FE_resids", "FE_resids", "y_minus_FEt_resids", "FEt_resids", "y_minus_RE_resids", "RE_resids"]
    ),
    "dif_mse.csv": (
        [FE_dif_mse, FE_dif_mae, FE_dif_mpe, FE_dif_mape, FEt_dif_mse, FEt_dif_mae, FEt_dif_mpe, FEt_dif_mape, RE_dif_mse, RE_dif_mae, RE_dif_mpe, RE_dif_mape],
        ["FE_mse", "FE_mae", "FE_mpe", "FE_mape", "FEt_mse", "FEt_mae", "FEt_mpe", "FEt_mape", "RE_mse", "RE_mae", "RE_mpe", "RE_mape"]
    ),
    "dif_mse_x1.csv": (
        [FE_dif_mse_x1, FE_dif_mae_x1, FE_dif_mpe_x1, FE_dif_mape_x1, FEt_dif_mse_x1, FEt_dif_mae_x1, FEt_dif_mpe_x1, FEt_dif_mape_x1, RE_dif_mse_x1, RE_dif_mae_x1, RE_dif_mpe_x1, RE_dif_mape_x1],
        ["FE_mse_x1", "FE_mae_x1", "FE_mpe_x1", "FE_mape_x1", "FEt_mse_x1", "FEt_mae_x1", "FEt_mpe_x1", "FEt_mape_x1", "RE_mse_x1", "RE_mae_x1", "RE_mpe_x1", "RE_mape_x1"]
    ),
    "dif_betas.csv": (
        [FE_dif_b0, FE_dif_b1, FE_dif_b2, FE_dif_b3, FEt_dif_b0, FEt_dif_b1, FEt_dif_b2, FEt_dif_b3, RE_dif_b0, RE_dif_b1, RE_dif_b2, RE_dif_b3],
        ["FE_b0", "FE_b1", "FE_b2", "FE_b3", "FEt_b0", "FEt_b1", "FEt_b2", "FEt_b3", "RE_b0", "RE_b1", "RE_b2", "RE_b3"]
    ),
    "dif_SEs.csv": (
        [FE_dif_b0SE, FE_dif_b1SE, FE_dif_b2SE, FE_dif_b3SE, FEt_dif_b0SE, FEt_dif_b1SE, FEt_dif_b2SE, FEt_dif_b3SE, RE_dif_b0SE, RE_dif_b1SE, RE_dif_b2SE, RE_dif_b3SE],
        ["FE_b0_SE", "FE_b1_SE", "FE_b2_SE", "FE_b3_SE", "FEt_b0_SE", "FEt_b1_SE", "FEt_b2_SE", "FEt_b3_SE", "RE_b0_SE", "RE_b1_SE", "RE_b2_SE", "RE_b3_SE"]
    ),
    "dif_Vars.csv": (
        [FE_dif_b0Var, FE_dif_b1Var, FE_dif_b2Var, FE_dif_b3Var, FEt_dif_b0Var, FEt_dif_b1Var, FEt_dif_b2Var, FEt_dif_b3Var, RE_dif_b0Var, RE_dif_b1Var, RE_dif_b2Var, RE_dif_b3Var],
        ["FE_b0_Var", "FE_b1_Var", "FE_b2_Var", "FE_b3_Var", "FEt_b0_Var", "FEt_b1_Var", "FEt_b2_Var", "FEt_b3_Var", "RE_b0_Var", "RE_b1_Var", "RE_b2_Var", "RE_b3_Var"]
    ),
    "dif_aic.csv": (
        [FE_dif_aic, FEt_dif_aic, RE_dif_aic],
        ["FE_aic", "FEt_aic", "RE_aic"]
    ),
    "dif_CIs.csv": (
        [CIlo_FE_dif_b0, CIhi_FE_dif_b0, CIlo_FE_dif_b1, CIhi_FE_dif_b1, CIlo_FE_dif_b2, CIhi_FE_dif_b2, CIlo_FE_dif_b3, CIhi_FE_dif_b3,
            CIlo_FEt_dif_b0, CIhi_FEt_dif_b0, CIlo_FEt_dif_b1, CIhi_FEt_dif_b1, CIlo_FEt_dif_b2, CIhi_FEt_dif_b2, CIlo_FEt_dif_b3, CIhi_FEt_dif_b3,
            CIlo_RE_dif_b0, CIhi_RE_dif_b0, CIlo_RE_dif_b1, CIhi_RE_dif_b1, CIlo_RE_dif_b2, CIhi_RE_dif_b2, CIlo_RE_dif_b3, CIhi_RE_dif_b3],
        ["FE_b0_CI_lo", "FE_b0_CI_hi","FE_b1_CI_lo", "FE_b1_CI_hi","FE_b2_CI_lo", "FE_b2_CI_hi","FE_b3_CI_lo", "FE_b3_CI_hi",
            "FEt_b0_CI_lo", "FEt_b0_CI_hi","FEt_b1_CI_lo", "FEt_b1_CI_hi","FEt_b2_CI_lo", "FEt_b2_CI_hi","FEt_b3_CI_lo", "FEt_b3_CI_hi",
            "RE_b0_CI_lo", "RE_b0_CI_hi","RE_b1_CI_lo", "RE_b1_CI_hi","RE_b2_CI_lo", "RE_b2_CI_hi","RE_b3_CI_lo", "RE_b3_CI_hi",]
    ),

}

for filename, (arrays, headers) in arrays_to_export.items():
    data = np.column_stack([as_1d(a) for a in arrays])
    df = pd.DataFrame(data, columns=headers)
    df.to_csv(filepath+"/"+filename, index=False)
    #df.to_csv(filepath+"Data/Case"+str(case)+"/"+filename, index=False)
