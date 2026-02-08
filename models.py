# ============================================================================
# Hierarchical Models for Nested Data
# ============================================================================
# Three approaches to modeling hierarchical/nested data:
#   - HierarchicalRandomForest (tree-based)
#   - HierarchicalNeuralNetwork (neural)
#   - HierarchicalMixedModel (statistical)
# ============================================================================

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError
from statsmodels.regression.mixed_linear_model import MixedLM
from torch.utils.data import DataLoader, TensorDataset
import multiprocessing


# ============================================================================
# TREE-BASED: Hierarchical Random Forest
# ============================================================================

class HierarchicalRandomForest:
    """
    Random forest that models residuals at each hierarchical level.
    
    The idea: fit a patient-level model first, then model the hospital-level
    residuals, then the region-level residuals. Predictions combine all levels.
    """
    
    def __init__(self, n_jobs=-1):
        self.n_jobs = n_jobs if n_jobs > 0 else multiprocessing.cpu_count()
        
        # Each level gets its own RF with different complexity
        self.patient_rf = RandomForestRegressor(
            n_estimators=100, max_depth=15, min_samples_leaf=5,
            random_state=42, n_jobs=self.n_jobs
        )
        self.hospital_rf = RandomForestRegressor(
            n_estimators=75, max_depth=12, min_samples_leaf=8,
            random_state=42, n_jobs=self.n_jobs
        )
        self.region_rf = RandomForestRegressor(
            n_estimators=50, max_depth=10, min_samples_leaf=10,
            random_state=42, n_jobs=self.n_jobs
        )
        
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, X, y, groups):
        """
        Fit models at each level of the hierarchy.
        
        groups should have keys: 'HOSP_NIS' (hospital) and 'HOSP_REGION' (region)
        """
        X_scaled = self.scaler.fit_transform(X)
        
        # Level 1: Patient features only
        self.patient_rf.fit(X_scaled, y)
        patient_pred = self.patient_rf.predict(X_scaled)
        
        # Level 2: Model hospital-level residuals
        hospital_residuals = y - patient_pred
        hospital_features = np.column_stack([X_scaled, groups['HOSP_NIS']])
        self.hospital_rf.fit(hospital_features, hospital_residuals)
        hospital_pred = self.hospital_rf.predict(hospital_features)
        
        # Level 3: Model region-level residuals
        region_residuals = hospital_residuals - hospital_pred
        region_features = np.column_stack([X_scaled, groups['HOSP_REGION']])
        self.region_rf.fit(region_features, region_residuals)
        
        self.is_fitted = True
        return self
    
    def predict(self, X, groups):
        if not self.is_fitted:
            raise NotFittedError("Call fit() first")
        
        X_scaled = self.scaler.transform(X)
        
        # Add predictions from all levels
        patient_pred = self.patient_rf.predict(X_scaled)
        hospital_pred = self.hospital_rf.predict(
            np.column_stack([X_scaled, groups['HOSP_NIS']])
        )
        region_pred = self.region_rf.predict(
            np.column_stack([X_scaled, groups['HOSP_REGION']])
        )
        
        return patient_pred + hospital_pred + region_pred


# ============================================================================
# NEURAL: Hierarchical Neural Network
# ============================================================================

class _HierarchicalNN(nn.Module):
    """
    Neural network with learned embeddings for each hierarchical level.
    
    Hospital and region get their own embedding tables. These are concatenated
    with patient features and fed through a shared prediction network.
    """
    
    def __init__(self, input_dim, num_hospitals, num_regions):
        super().__init__()
        
        self.hospital_dim = 16
        self.region_dim = 8
        self.feature_dim = 32
        
        # Embeddings for each group level
        self.hospital_emb = nn.Embedding(num_hospitals, self.hospital_dim)
        self.region_emb = nn.Embedding(num_regions, self.region_dim)
        nn.init.xavier_uniform_(self.hospital_emb.weight)
        nn.init.xavier_uniform_(self.region_emb.weight)
        
        # Process patient features
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Process hospital embeddings
        self.hospital_net = nn.Sequential(
            nn.Linear(self.hospital_dim, self.hospital_dim),
            nn.LayerNorm(self.hospital_dim),
            nn.ReLU()
        )
        
        # Process region embeddings
        self.region_net = nn.Sequential(
            nn.Linear(self.region_dim, self.region_dim),
            nn.LayerNorm(self.region_dim),
            nn.ReLU()
        )
        
        # Final prediction network
        combined_dim = self.feature_dim + self.hospital_dim + self.region_dim
        self.predictor = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x, hospital_idx, region_idx):
        # Get embeddings
        hospital_emb = self.hospital_net(self.hospital_emb(hospital_idx))
        region_emb = self.region_net(self.region_emb(region_idx))
        patient_features = self.feature_net(x)
        
        # Combine and predict
        combined = torch.cat([patient_features, hospital_emb, region_emb], dim=1)
        return self.predictor(combined).squeeze()


class HierarchicalNeuralNetwork:
    """
    Wrapper class for training and inference with the hierarchical NN.
    """
    
    def __init__(self, input_dim, num_hospitals, num_regions):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_dim = input_dim
        self.num_hospitals = num_hospitals
        self.num_regions = num_regions
        
        # Training params
        self.learning_rate = 0.0001
        self.max_epochs = 30
        self.patience = 5
        self.batch_size = 64
        
        self.model = None
    
    def fit(self, X, y, groups):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.model = _HierarchicalNN(
            self.input_dim, self.num_hospitals, self.num_regions
        ).to(self.device)
        
        # Prep data
        dataset = TensorDataset(
            torch.FloatTensor(X),
            torch.LongTensor(groups['HOSP_NIS']),
            torch.LongTensor(groups['HOSP_REGION']),
            torch.FloatTensor(y)
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=3, factor=0.5
        )
        criterion = nn.MSELoss()
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.max_epochs):
            self.model.train()
            epoch_loss = 0
            
            for x_batch, h_batch, r_batch, y_batch in loader:
                x_batch = x_batch.to(self.device)
                h_batch = h_batch.to(self.device)
                r_batch = r_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                pred = self.model(x_batch, h_batch, r_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(loader)
            scheduler.step(avg_loss)
            
            print(f"Epoch {epoch}: loss = {avg_loss:.4f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print("Early stopping")
                    break
        
        return self
    
    def predict(self, X, groups):
        self.model.eval()
        
        dataset = TensorDataset(
            torch.FloatTensor(X),
            torch.LongTensor(groups['HOSP_NIS']),
            torch.LongTensor(groups['HOSP_REGION'])
        )
        loader = DataLoader(dataset, batch_size=self.batch_size * 2)
        
        predictions = []
        with torch.no_grad():
            for x_batch, h_batch, r_batch in loader:
                x_batch = x_batch.to(self.device)
                h_batch = h_batch.to(self.device)
                r_batch = r_batch.to(self.device)
                
                pred = self.model(x_batch, h_batch, r_batch)
                predictions.append(pred.cpu())
        
        return torch.cat(predictions).numpy()


# ============================================================================
# STATISTICAL: Hierarchical Mixed Model
# ============================================================================

class HierarchicalMixedModel:
    """
    Linear mixed model with random effects for hospital.
    
    This is the classic statistical approach — fixed effects for patient
    features, random intercepts for hospitals. Interpretable but can be
    slow to converge on large datasets.
    """
    
    def __init__(self, max_iter=100):
        self.scaler = StandardScaler()
        self.model = None
        self.max_iter = max_iter
    
    def fit(self, X, y, groups):
        X_scaled = self.scaler.fit_transform(X)
        
        # Add intercept
        X_with_intercept = np.column_stack([np.ones(len(X_scaled)), X_scaled])
        
        # Hospital as grouping variable
        hospital_groups = groups['HOSP_NIS']
        
        model = MixedLM(endog=y, exog=X_with_intercept, groups=hospital_groups)
        
        # Try different optimization methods if one fails
        try:
            self.model = model.fit(maxiter=self.max_iter, method='lbfgs')
        except:
            try:
                self.model = model.fit(maxiter=self.max_iter, method='cg')
            except:
                # Last resort: simpler model
                self.model = model.fit(maxiter=self.max_iter // 2, reml=False)
        
        return self
    
    def predict(self, X):
        if self.model is None:
            raise RuntimeError("Call fit() first")
        
        X_scaled = self.scaler.transform(X)
        X_with_intercept = np.column_stack([np.ones(len(X_scaled)), X_scaled])
        
        return self.model.predict(X_with_intercept)
    
    def get_params(self):
        """Return estimated coefficients."""
        if self.model is None:
            return None
        return self.model.params


# ============================================================================
# Quick test
# ============================================================================

if __name__ == "__main__":
    # Generate some fake hierarchical data
    np.random.seed(42)
    n = 1000
    
    X = np.random.randn(n, 10)
    hospitals = np.random.randint(0, 50, n)
    regions = np.random.randint(0, 4, n)
    
    # Target with hierarchical structure
    y = X[:, 0] * 2 + hospitals * 0.1 + regions * 0.5 + np.random.randn(n)
    
    groups = {'HOSP_NIS': hospitals, 'HOSP_REGION': regions}
    
    # Test each model
    print("Testing HierarchicalRandomForest...")
    hrf = HierarchicalRandomForest()
    hrf.fit(X[:800], y[:800], {k: v[:800] for k, v in groups.items()})
    pred_hrf = hrf.predict(X[800:], {k: v[800:] for k, v in groups.items()})
    print(f"  R² = {1 - np.var(y[800:] - pred_hrf) / np.var(y[800:]):.3f}")
    
    print("\nTesting HierarchicalMixedModel...")
    hmm = HierarchicalMixedModel()
    hmm.fit(X[:800], y[:800], {k: v[:800] for k, v in groups.items()})
    pred_hmm = hmm.predict(X[800:])
    print(f"  R² = {1 - np.var(y[800:] - pred_hmm) / np.var(y[800:]):.3f}")
    
    print("\nDone!")
