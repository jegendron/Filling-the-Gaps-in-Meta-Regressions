## Filling-the-Gaps-in-Meta-Regressions

>Note: This repository corresponds to Chapter 4 of the following dissertation:

### What the R Code Does
- Runs two simulations generating data with: 1) **study** heterogeneity 2) joint heterogeneity in **location** and **time**
- Tests nine imputation approaches: three parametric, two tree-based, and four deep-learning
- Calculates estimator bias, MSE, power, and CI coverage
- Generates figures summarizing results

### Key Workflow
- Vary number of countries: 5, 15, 33
- Vary time heterogeneity and location heterogeneity
- Repeat simulations via Monte Carlo design

### Metrics Output
- Power, MSE, bias, confidence intervals, coverage rates
- Exported as data frames for comparison

### Sample Code

This follows the same design of our previous work, but the key difference is in the data generation (https://github.com/jegendron/Evaluating-Meta-Regression-Techniques-A-Simulation-Study-on-Heterogeneity-in-Location-and-Time)

Note that the full code is attached separately, and although Simulation 1 and 2 have a lot of overlap, they are shown separately due to their different groupings.

#### Data Generation - Simulation 1 (Joint Location-Time Heterogeneity)
```
import time
import numpy as np

### Imputation libraries
from statsmodels.imputation.mice import MICEData           # for MI & LH
import statsmodels.api as sm                               # for LH

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

    ###### COUNTRY 1 ######
    
    ### 2020 ###
    mu_y = spreadMuY
    mu_0 = np.array([mu_y, mu_x1, mu_x2, mu_x3])  # The mean for y changes, x mean remains
    
    z_data = rng.multivariate_normal(mean=mu_0, cov=SIGMA, size=n)
    #z_data = np.random.multivariate_normal(mean=mu_0, cov=SIGMA, size=n)  # z acts as a temporary placeholder
    y = z_data[:, 0]
    x = z_data[:, 1:4]

    X = pd.DataFrame({
        'constant': 1,
        'x.1': x[:, 0],
        'x.2': x[:, 1],
        'x.3': x[:, 2],
        'year': 2020,
        'country': 1,
        'paper': 1,
        'trend': -1
    })
    
    ### 2021-2024 ###
    
    # 2020 is not in the loop since it's the starting point
    
    for i in range(1, numYears):
        mu_y = mu_y + (timeHet * i)  # The mean by _ per year
        mu_0 = np.array([mu_y, mu_x1, mu_x2, mu_x3])  # x mean remains
        
        z_data = rng.multivariate_normal(mean=mu_0, cov=SIGMA, size=n)
        #z_data = np.random.multivariate_normal(mean=mu_0, cov=SIGMA, size=n)
        y2 = z_data[:, 0]
        x = z_data[:, 1:4]

        X2 = pd.DataFrame({
            'constant': 1,
            'x.1': x[:, 0],
            'x.2': x[:, 1],
            'x.3': x[:, 2],
            'year': i + 2020,
            'country': 1,
            'paper': i + 1,
            'trend': -1 + i / 2
        })
        
        X = pd.concat([X, X2], ignore_index=True)
        y = np.concatenate([y, y2])

    
    
    ###### COUNTRY 2 - 15 ######
    # Country 1 is not in the loop since it's the starting point
    
    ### 2020-2024 ###
    paper_iterator = numYears + 1  # since Country 1 includes 1 paper per year
    countryIndex = 2
    
    if numCountries == 15 and spreadMuY == -2:
        temp = [-1.66, -1.33, -1, -0.66, -0.33, -0.15, 0, 0.15, 0.33, 0.66, 1, 1.33, 1.66, 2]  # for calculating spreadMuY
        for i in range(len(temp)):
            for j in range(numYears):
                mu_y = temp[i] + (timeHet * j)  # The mean for y increases by 0.5 per country, also increases by _ per year
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
                    'country': i + 2,
                    'paper': paper_iterator,
                    'trend': -1 + j / 2
                })
                
                X = pd.concat([X, X2], ignore_index=True)
                y = np.concatenate([y, y2])
                paper_iterator = paper_iterator + 1
            countryIndex = countryIndex + 1
    
    if numCountries == 15 and spreadMuY == -10:
        temp = [-8.5, -7, -5.5, -4, -2.5, -1, 0, 1, 2.5, 4, 5.5, 7, 8.5, 10]  # for calculating spreadMuY
        for i in range(len(temp)):
            for j in range(numYears):
                mu_y = temp[i] + (timeHet * j)  # The mean for y increases by 0.5 per country, also increases by _ per year
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
                    'country': i,
                    'paper': paper_iterator,
                    'trend': -1 + j / 2
                })
            
                X = pd.concat([X, X2], ignore_index=True)
                y = np.concatenate([y, y2])
                paper_iterator = paper_iterator + 1
            countryIndex = countryIndex + 1



    #################################
    ### MISSING DATA CONSTRUCTION ###
    #################################  
         
    # Random
    if case < 5:
        # For every n observations, randomly make 6.6% of them missing
        
        rows = list(range(len(X)))
        blocks = {}
        for row in rows:
            block_id = math.ceil((row + 1) / n)  # Group rows into blocks
            if block_id not in blocks:
                blocks[block_id] = []
            blocks[block_id].append(row)
        
        missing_rows = []
        
        for b in blocks.values():
            k0 = min(6.6, len(b))  # 6.6 rows per 100
            missing_rows.extend(random.sample(b, int(k0)))
        
        X.loc[missing_rows, 'x.1'] = np.nan
    
    elif case > 16 and case < 21:
        # For every 100 rows, randomly make 20% of them missing
        
        rows = list(range(len(X)))
        blocks = {}
        for row in rows:
            block_id = math.ceil((row + 1) / n)  # Group rows into blocks
            if block_id not in blocks:
                blocks[block_id] = []
            blocks[block_id].append(row)
        
        missing_rows = []
        
        for b in blocks.values():
            k0 = min(20, len(b))  # 20 rows per 100
            missing_rows.extend(random.sample(b, int(k0)))
        
        X.loc[missing_rows, 'x.1'] = np.nan
    
    # SMALL Location-time (2 years AND 2 locations == 2020-2021 & Locations 1-2)
    elif case > 4 and case < 9:
        X.loc[(X['year'] < 2022) | (X['country'] < 3), 'x.1'] = np.nan
    # LARGE Location-time (2 years AND 7 locations == 2020-2021 & Locations 1-7)
    elif case > 20 and case < 25:
        X.loc[(X['year'] < 2022) | (X['country'] < 8), 'x.1'] = np.nan
    
    # SMALL Location (1 location == Location 1)
    elif case > 8 and case < 13:
        X.loc[X['country'] == 1, 'x.1'] = np.nan
    # LARGE Location (3 locations == Locations 1-3)
    elif case > 24 and case < 29:
        X.loc[X['country'] < 4, 'x.1'] = np.nan
    
    # SMALL time (1 year == 2020)
    elif case > 12 and case < 17:
        X.loc[X['year'] == 2020, 'x.1'] = np.nan
    # LARGE time (2 years == 2020-2021)
    elif case > 28 and case < 33:
        X.loc[X['year'] < 2022, 'x.1'] = np.nan    
    

    
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

        ###n_epochs = 200
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
            ###latent_dim=10,
            ###n_importance_samples=20
            latent_dim=20,
            n_importance_samples=10
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
```



#### Data Generation - Simulation 2 (Study Heterogeneity)
```
import time
import numpy as np

### Imputation libraries
from statsmodels.imputation.mice import MICEData           # for MI & LH
import statsmodels.api as sm                               # for LH

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
```
