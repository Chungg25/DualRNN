import torch
import torch.nn as nn
from layers.RevIN import RevIN
from layers.PatchTST_layers import series_decomp

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        
        # Các tham số cơ bản
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        
        # Tham số cho patching
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.patch_num = self.seq_len // self.patch_len
        
        # Decomposition
        self.kernel_size = configs.kernel_size
        self.decomp = series_decomp(self.kernel_size)
        
        # RevIN normalization
        self.revin = configs.revin
        if self.revin:
            self.revin_layer = RevIN(self.enc_in, affine=True, subtract_last=False)
            
        # Trend Model
        self.trend_model = TrendModel(
            d_model=self.d_model,
            patch_len=self.patch_len,
            stride=self.stride,
            dropout=self.dropout
        )
        
        # Seasonal Model
        self.seasonal_model = SeasonalModel(
            d_model=self.d_model,
            patch_len=self.patch_len,
            stride=self.stride,
            dropout=self.dropout
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(self.patch_len * 2, self.d_model),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # Output projection
        self.projection = nn.Linear(self.d_model, self.pred_len * self.enc_in)
        
    def forward(self, x):
        # x: [batch_size, seq_len, n_vars]
        batch_size = x.size(0)
        
        # 1. RevIN normalization
        if self.revin:
            x = self.revin_layer(x, 'norm')
            
        # 2. Decomposition
        trend, seasonal = self.decomp(x)
        
        # 3. Process trend and seasonal separately
        trend = trend.contiguous()
        seasonal = seasonal.contiguous()
        
        trend_out = self.trend_model(trend)  # [batch, patch_len]
        seasonal_out = self.seasonal_model(seasonal)  # [batch, patch_len]
        
        # 4. Fusion
        combined = torch.cat([trend_out, seasonal_out], dim=-1)  # [batch, patch_len*2]
        y = self.fusion(combined)  # [batch, d_model]
        
        # 5. Project to prediction length
        y = self.projection(y)  # [batch, pred_len * enc_in]
        y = y.reshape(batch_size, self.pred_len, self.enc_in)  # [batch, pred_len, enc_in]
        
        # 6. Denormalization
        if self.revin:
            y = self.revin_layer(y, 'denorm')
            
        return y

class TrendModel(nn.Module):
    def __init__(self, d_model, patch_len, stride, dropout):
        super(TrendModel, self).__init__()
        
        self.patch_len = patch_len
        self.stride = stride
        
        # Patch embedding
        self.patch_embedding = nn.Sequential(
            nn.Linear(patch_len, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Trend processing
        self.trend_processor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Prediction head
        self.predict = nn.Sequential(
            nn.Linear(d_model, patch_len),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # x: [batch_size, seq_len, n_vars]
        batch_size, seq_len, n_vars = x.shape
        
        # Patching
        x = x.permute(0, 2, 1).contiguous()  # [batch, n_vars, seq_len]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # [batch, n_vars, num_patches, patch_len]
        
        # Reshape và make contiguous
        num_patches = x.size(2)
        x = x.reshape(batch_size * n_vars * num_patches, self.patch_len).contiguous()
        
        # Patch embedding
        x = self.patch_embedding(x)  # [batch*n_vars*num_patches, d_model]
        
        # Process trend
        x = self.trend_processor(x)
        
        # Reshape back
        x = x.reshape(batch_size, n_vars * num_patches, -1).contiguous()
        
        # Global pooling
        x = x.mean(dim=1)  # [batch, d_model]
        
        # Prediction
        y = self.predict(x)  # [batch, patch_len]
        
        return y

class SeasonalModel(nn.Module):
    def __init__(self, d_model, patch_len, stride, dropout):
        super(SeasonalModel, self).__init__()
        
        self.patch_len = patch_len
        self.stride = stride
        
        # Patch embedding
        self.patch_embedding = nn.Sequential(
            nn.Linear(patch_len, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Seasonal processing
        self.seasonal_processor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Prediction head
        self.predict = nn.Sequential(
            nn.Linear(d_model, patch_len),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # x: [batch_size, seq_len, n_vars]
        batch_size, seq_len, n_vars = x.shape
        
        # Patching
        x = x.permute(0, 2, 1).contiguous()  # [batch, n_vars, seq_len]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # [batch, n_vars, num_patches, patch_len]
        
        # Reshape và make contiguous
        num_patches = x.size(2)
        x = x.reshape(batch_size * n_vars * num_patches, self.patch_len).contiguous()
        
        # Patch embedding
        x = self.patch_embedding(x)  # [batch*n_vars*num_patches, d_model]
        
        # Process seasonal
        x = self.seasonal_processor(x)
        
        # Reshape back
        x = x.reshape(batch_size, n_vars * num_patches, -1).contiguous()
        
        # Global pooling
        x = x.mean(dim=1)  # [batch, d_model]
        
        # Prediction
        y = self.predict(x)  # [batch, patch_len]
        
        return y