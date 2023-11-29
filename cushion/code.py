########################################################################################################################
########################################################### START ######################################################

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter

from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from time import time

import torch
import torch.nn as nn
from torch.nn import Linear, LeakyReLU, BatchNorm1d
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from matplotlib.ticker import ScalarFormatter

class h2_cushion_rom(nn.Module):
    '''
    H2 Cushion Gas proxy model: 
        a torch dense nn to estimate [efft,ymft,gwrt,J] 
        from various geologic and operational parameters.
    '''
    def __init__(self, in_features, out_features, hidden_sizes=[128,64,16]):   # 100,60,10
        super().__init__()                                                     # Calls the constructor of the superclass
        assert len(hidden_sizes) >= 1 , 'specify at least one hidden layer'
        layers = nn.ModuleList()
        layer_sizes = [in_features] + hidden_sizes
        
        for dim_in, dim_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(Linear(dim_in, dim_out))                             # fully-connected
            layers.append(BatchNorm1d(dim_out))                                # batch normalization
            layers.append(LeakyReLU(0.2))                                      # activation
                                                                                 
        self.layers = nn.Sequential(*layers)                                   # serialize
        self.out_layer = Linear(hidden_sizes[-1], out_features)                # output layer
     
    def forward(self, x):
        out = x.view(x.shape[0], -1)
        out = self.layers(out)
        out = self.out_layer(out)
        return out

class Custom_Loss(nn.Module):
    '''
    L1 (MAE) loss function.
    '''
    def __init__(self):
        super(Custom_Loss, self).__init__()
    def forward(self, pred, true):
        mae_loss = nn.L1Loss()(pred, true)  # L1 loss, which is MAE
        return mae_loss

class Dual_Loss(nn.Module):
    '''
    custom dual loss function with 
    weighted average combination of (smooth) L1 and L2 metrics.
    '''
    def __init__(self):
        super(Dual_Loss, self).__init__()
        self.l2_weight = 0.1
    def forward(self, pred, true):
        l1_loss = nn.MSELoss()(pred, true)                                     # l2 loss
        l2_loss = nn.SmoothL1Loss()(pred, true)                                # l1 loss (smooth - Huber/L1)
        total_loss = (1-self.l2_weight)*l1_loss + self.l2_weight*l2_loss
        return total_loss

class CustomScalarFormatter(ScalarFormatter):
    def _set_format(self, vmin=None, vmax=None):
        self.format = "%1.2f"
        
class H2Toolkit:
    '''
    a large module for self-contained for:
    (1) data loading and processing,
    (2) defining the proxy model, training, and predictions,
    (3) computing performance metrics and plotting results.
    '''
    def __init__(self):
        self.return_data = False                                        # return data?
        self.return_plot = True                                         # print plots?
        self.save_data   = True                                         # save data?
        self.verbose     = True                                         # print outputs?
        self.inp         = 12                                           # n features
        self.out         = 4                                            # n targets
        self.xcols       = range(12)                                    # feature columns
        self.ycols       = [12, 14, 15, 17]                             # target columns
        self.noise_flag  = False                                        # add noise?
        self.noise       = [0, 0.05]                                    # added noise mean, std
        self.gwrt_cutoff = 1e5                                          # GWRT outlier threshold
        self.std_outlier = 3                                            # target outlier threshold
        self.y_labels    = ['efft','ymft','gwrt','injt']                # target names
        self.test_size   = 0.15                                         # Test size
        self.valid_size  = 0.17647                                      # Train-validation split size
        self.epochs      = 200                                          # training epochs
        self.batch_size  = 512                                          # batch size
        self.delta_epoch = 10                                           # print performance every n epochs
        self.metrics     = {'training_loss':[], 'validation_loss':[]}   # pre-load metrics dictionary


    def check_torch_gpu(self):
        '''
        Check torch build in python to ensure GPU is available for training.
        '''
        torch_version, cuda_avail = torch.__version__, torch.cuda.is_available()
        count, name = torch.cuda.device_count(), torch.cuda.get_device_name()
        #py_version, conda_env_name = sys.version, sys.executable.split('\\')[-2]
        if self.verbose:
            print('\n-------------------------------------------------')
            print('------------------ VERSION INFO -----------------')
            #print('Conda Environment: {} | Python version: {}'.format(conda_env_name, py_version))
            print('Torch version: {}'.format(torch_version))
            print('Torch build with CUDA? {}'.format(cuda_avail))
            print('# Device(s) available: {}, Name(s): {}'.format(count, name))
        self.device = torch.device('cuda' if cuda_avail else 'cpu')
        return None


    def train(self):
        def count_params(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Model, Optimizer and Loss Function
        self.model     = h2_cushion_rom(self.inp, self.out, hidden_sizes=[1024,512,256,128,64]).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=2e-3)
        self.loss_fn   = Custom_Loss()
        
        # Split into test, training, and validation sets
        self.X_train_val, self.X_test, self.y_train_val, self.y_test = train_test_split(self.X_dataset,self.y_dataset,test_size=self.test_size)  # Hold out test set
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train_val, self.y_train_val, test_size = self.valid_size)   # Split training and validation sets    
    
        if self.verbose:
            num_data = self.X_train.shape[0] + self.X_val.shape[0] + self.X_test.shape[0]
            print('\n---------------- PROCESSED DATA INFORMATION ---------------')
            print('TRAIN: {} | VALIDATION: {} | TEST: {}'.format(self.X_train.shape[0], self.X_val.shape[0], self.X_test.shape[0]))
            print('TRAIN: {:.3f} | VALIDATION: {:.3f} | TEST: {:.3f}'.format(self.X_train.shape[0]/num_data, self.X_val.shape[0]/num_data, self.X_test.shape[0]/num_data))
        
        # Save the unscaled train, validation, and test data
        np.save('data/X_train.npy', self.X_train)
        np.save('data/y_train.npy', self.y_train)
        
        np.save('data/X_val.npy', self.X_val)
        np.save('data/y_val.npy', self.y_val)
        
        np.save('data/X_test.npy', self.X_test)
        np.save('data/y_test.npy', self.y_test)        
    
        # Tensorize and Move to GPU
        self.X_test_tensor  = torch.Tensor(self.X_test).to(self.device)        # Tensorize X_test  to gpu      
        self.X_train_tensor = torch.Tensor(self.X_train).to(self.device)       # Tensorize X_train to gpu
        self.y_train_tensor = torch.Tensor(self.y_train).to(self.device)       # Tensorize y_train to gpu
        self.X_valid_tensor = torch.Tensor(self.X_val).to(self.device)         # Tensorize X_val   to gpu
        self.y_valid_tensor = torch.Tensor(self.y_val).to(self.device)         # Tensorize y_val   to gpu        
    
        # Create Dataloaders
        train_dataset = TensorDataset(self.X_train_tensor, self.y_train_tensor)
        valid_dataset = TensorDataset(self.X_valid_tensor, self.y_valid_tensor)
        train_loader  = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        valid_loader  = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=True)
    
        # Initialize with a high value
        best_val_loss = float('inf')  # Initialize with a high value
    
        print('\n----------------- MODEL TRAINING ----------------')
        print('Number of trainable parameters: {:,}'.format(count_params(self.model)))
        # Training loop
        for epoch in range(self.epochs):
            train_loss_avg = self._train_one_epoch(train_loader)
            val_loss_avg   = self._validate_one_epoch(valid_loader)
            
            # Save losses to metrics
            self.metrics['training_loss'].append(train_loss_avg)
            self.metrics['validation_loss'].append(val_loss_avg)
    
            # Save the best model based on validation loss
            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                torch.save(self.model.state_dict(), 'best_model.pt')  
                example_input = torch.randn(1, self.inp)
                example_input = example_input.to(self.device)
                traced_model = torch.jit.trace(self.model, example_input)
                torch.jit.save(traced_model, 'traced_model.pt')
    
            # Print loss every self.delta_epoch epochs
            if epoch % self.delta_epoch == 0:
                print(f'Epoch [{epoch}/{self.epochs}], Average Train Loss: {train_loss_avg:.5f}, Average Val Loss: {val_loss_avg:.5f}')


    def _train_one_epoch(self, train_loader):                                    
        self.model.train()  # Switch to training mode
        total_loss = 0.0
        for inp, true in train_loader:
            self.optimizer.zero_grad()  # Zero gradients                       # Sets the model to training mode 
            pred = self.model(inp)                                                
            loss = self.loss_fn(pred, true)                                       
            loss.backward()                                                    # Zero out any gradients from the previous iteration
            self.optimizer.step()                                              # Forward pass: Compute predicted outputs by passing inputs to the model          
            total_loss += loss.item() * inp.size(0)                            # Calculate the loss using the predicted outputs and true labels
        return total_loss / len(train_loader.dataset)                          # Backward pass: Compute gradient of the loss with respect to model parameters
                                                                               # Optimize the model: Update the model parameters using computed gradients  
                                                                               # Accumulate the total loss for the epoch 
    def _validate_one_epoch(self, valid_loader):                   
        self.model.eval()  # Switch to evaluation mode                         # Switch the model to evaluation mode
        total_loss = 0.0                                                       # Initialize a variable to accumulate the total validation loss across all mini-batches.
        with torch.no_grad():                                                  # Use the no_grad context to ensure no gradients are calculated, which saves memory and speeds up computation. 
            for inp, true in valid_loader:                                     # Loop over the validation data in mini-batches provided by valid_loader
                pred = self.model(inp)                                         # Pass the input data through the model to get predictions
                loss = self.loss_fn(pred, true)                                # Compute the loss for the current mini-batch by comparing predictions to true labels
                total_loss += loss.item() * inp.size(0)                        # Multiply the loss by the mini-batch size and accumulate it to the total validation loss
        return total_loss / len(valid_loader.dataset)  


    def evaluate(self):
        '''
        Compute and print performance metrics: R^2, MSE, MAE. 
        This is done for the overall dataset, as well as per-target basis.
        '''
        # Load the best model for final evaluations and predictions
        self.model.load_state_dict(torch.load('best_model.pt'))

        # Predict and de-tensorize 
        self.y_train_pred = self.model(self.X_train_tensor).cpu().detach().numpy()
        self.y_val_pred   = self.model(self.X_valid_tensor).cpu().detach().numpy()
        self.y_test_pred  = self.model(self.X_test_tensor).cpu().detach().numpy()
        
        # Inverse transform 
        self.y_train_inv = self.y_scaler.inverse_transform(self.y_train)
        self.y_val_inv   = self.y_scaler.inverse_transform(self.y_val)  
        self.y_test_inv  = self.y_scaler.inverse_transform(self.y_test) 
        
        self.y_train_pred_inv = self.y_scaler.inverse_transform(self.y_train_pred)
        self.y_val_pred_inv   = self.y_scaler.inverse_transform(self.y_val_pred) 
        self.y_test_pred_inv  = self.y_scaler.inverse_transform(self.y_test_pred)
        
        # Additional transformation for 'gwrt' column
        self.y_train_inv[:, 2] = 10 ** self.y_train_inv[:, 2]
        self.y_val_inv[:, 2]   = 10 ** self.y_val_inv[:, 2]
        self.y_test_inv[:, 2]  = 10 ** self.y_test_inv[:, 2]
        
        self.y_train_pred_inv[:, 2] = 10 ** self.y_train_pred_inv[:, 2]
        self.y_val_pred_inv[:, 2]   = 10 ** self.y_val_pred_inv[:, 2]
        self.y_test_pred_inv[:, 2]  = 10 ** self.y_test_pred_inv[:, 2]
        
        self.y_train_inv[:, 3] = 10 ** self.y_train_inv[:, 3]
        self.y_val_inv[:, 3]   = 10 ** self.y_val_inv[:, 3]
        self.y_test_inv[:, 3]  = 10 ** self.y_test_inv[:, 3]
        
        self.y_train_pred_inv[:, 3] = 10 ** self.y_train_pred_inv[:, 3]
        self.y_val_pred_inv[:, 3]   = 10 ** self.y_val_pred_inv[:, 3]
        self.y_test_pred_inv[:, 3]  = 10 ** self.y_test_pred_inv[:, 3]
             
        print('\n-------------- PERFORMANCE METRICS --------------')
        # Compute metrics for each output seperately
        for i in range(len(self.ycols)):
            name = self.y_labels[i]
            
            # Change the output names 
            if name == 'efft':
                name = 'E$_h$'
                unit = '/'
            elif name == 'ymft':
                name = 'P$_h$'
                unit = '/'
            elif name == 'gwrt':
                name = 'GWR'
                unit = '/'
            elif name == 'injt':
                name = 'J'
                unit = 'm$^3$/s/MPa'
                
            # Cutoff
            self.y_train_pred_inv[:,i] = np.clip(self.y_train_pred_inv[:,i], self.y_train_inv[:,i].min(), self.y_train_inv[:,i].max())
            self.y_val_pred_inv[:,i]   = np.clip(self.y_val_pred_inv[:,i]  , self.y_val_inv[:,i].min(),   self.y_val_inv[:,i].max()  )    
            self.y_test_pred_inv[:,i]  = np.clip(self.y_test_pred_inv[:,i] , self.y_test_inv[:,i].min(),  self.y_test_inv[:,i].max() )          
              
            # R^2
            r2_train_seperate = r2_score(self.y_train_inv[:,i], self.y_train_pred_inv[:,i])
            r2_val_seperate   = r2_score(self.y_val_inv[:,i], self.y_val_pred_inv[:,i])
            r2_test_seperate  = r2_score(self.y_test_inv[:,i], self.y_test_pred_inv[:,i])      
 
            # Print
            print('{} TRAIN: R2={:.5f}  | VALID: R2={:.5f}  | TEST: R2={:.5f}'.format(name, r2_train_seperate, r2_val_seperate, r2_test_seperate))
  
            # Plot true vs. predicted for each dataset
            self.plot_true_vs_predicted(self.y_train_inv[:,i], self.y_train_pred_inv[:,i], name + ' - Training',   name + '_Training'  , r2_train_seperate, unit )
            self.plot_true_vs_predicted(self.y_val_inv[:,i],   self.y_val_pred_inv[:,i],   name + ' - Validation', name + '_Validation', r2_val_seperate,   unit )
            self.plot_true_vs_predicted(self.y_test_inv[:,i],  self.y_test_pred_inv[:,i],  name + ' - Test',       name + '_Test'      , r2_test_seperate,  unit )


    def plot_true_vs_predicted(self, true, predicted, title, figure_name, r2_value, unit="", epsilon=1e-10):
        plt.figure(figsize=(10, 6))
        
        # Compute relative error
        relative_error = np.abs(predicted - true) / (np.abs(true) + epsilon)
        avg_relative_error = np.mean(relative_error)  # Average relative error
        
        # Define the ticks generation function 
        def custom_ticks(mn, mx):
            return [mn, mn + 0.25*(mx - mn), mn + 0.5*(mx - mn), mn + 0.75*(mx - mn), mx] 
    
        # Compute 2D histogram
        h = plt.hist2d(true, predicted, bins=100, cmap='turbo', norm=LogNorm())
        
        # Colorbar to represent the frequency or density of points
        cbar = plt.colorbar(h[3])
        cbar.set_label('Frequency', fontsize=22)
        cbar.ax.tick_params(labelsize=18)  # Set tick label font size for colorbar
        
        # Identity line
        plt.plot([min(true), max(true)], [min(true), max(true)], color='red', linestyle='--', linewidth=2.0)
        
        # Labels, title, and grid
        plt.xlabel(f'Truth ({unit})', fontsize=22)
        plt.ylabel(f'Prediction ({unit})', fontsize=22)
        plt.title(title, fontsize=22, fontweight='bold')
        
        # Add R^2 and average relative error annotations
        plt.annotate(f'R$^2$ = {r2_value:.3f}', 
                      xy=(0.05, 0.95), 
                      xycoords='axes fraction', 
                      fontsize=22, 
                      fontweight='bold', 
                      va='top')
        
        avg_rel_err_str = r'$\mathbf{\bar{\varepsilon}_r}$' + f'  = {avg_relative_error:.3f}'
        plt.annotate(avg_rel_err_str, 
                      xy=(0.05, 0.85),  
                      xycoords='axes fraction', 
                      fontsize=22, 
                      fontweight='bold', 
                      va='top')
        
        # Improve axis aesthetics
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#666666')
        ax.spines['left'].set_color('#666666')
        
        def format_tick_labels(x, pos):
            return f"{x:.2f}"
        
        # Use 2 decimal places for x and y-axis numbers
        formatter = FuncFormatter(format_tick_labels)
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
        
        # Set custom ticks for x and y axes
        ax.set_xticks(custom_ticks(min(true), max(true)))
        ax.set_yticks(custom_ticks(min(predicted), max(predicted)))
        
        # Set tick label font size
        ax.tick_params(labelsize=22)
        
        if max(ax.get_yticks()) > 100:
            formatter = CustomScalarFormatter(useMathText=True)
            formatter.set_scientific(True) 
            formatter.set_powerlimits((-1,1)) 
            ax.yaxis.set_major_formatter(formatter)
        
        # Set the exponent font size
        ax.yaxis.offsetText.set_fontsize(22)
        
        if max(ax.get_xticks()) > 100:
            formatter = CustomScalarFormatter(useMathText=True)
            formatter.set_scientific(True) 
            formatter.set_powerlimits((-1,1)) 
            ax.xaxis.set_major_formatter(formatter)
        
        # Set the exponent font size
        ax.xaxis.offsetText.set_fontsize(22)
        
        # Ensure 'figures' directory exists
        if not os.path.exists("figures"):
            os.makedirs("figures")   
        
        plt.tight_layout()
        plt.savefig(os.path.join("figures", figure_name + ".pdf"), format="pdf")
        plt.show()

    
    def read_data(self, n_subsample=None):
        '''
        Import data from the 4 main CSV files [ch4, co2, n2, nocg].
        '''
        data_sa_ch4   = pd.read_csv('data/data_sa_ch4.csv',   index_col=0)     # CH4  sa  cushion gas dataset
        data_sa_co2   = pd.read_csv('data/data_sa_co2.csv',   index_col=0)     # CO2  sa  cushion gas dataset
        data_sa_n2    = pd.read_csv('data/data_sa_n2.csv',    index_col=0)     # N2   sa  cushion gas dataset
        data_sa_nocg  = pd.read_csv('data/data_sa_nocg.csv',  index_col=0)     # NONE sa  cushion gas dataset 
        
        data_dgr_ch4  = pd.read_csv('data/data_dgr_ch4.csv',  index_col=0)     # CH4  dgr cushion gas dataset
        data_dgr_co2  = pd.read_csv('data/data_dgr_co2.csv',  index_col=0)     # CO2  dgr cushion gas dataset
        data_dgr_n2   = pd.read_csv('data/data_dgr_n2.csv',   index_col=0)     # N2   dgr cushion gas dataset
        data_dgr_nocg = pd.read_csv('data/data_dgr_nocg.csv', index_col=0)     # NONE dgr cushion gas dataset         
        
        data_all = pd.concat([data_sa_ch4,  data_sa_co2,  data_sa_n2,  data_sa_nocg, 
                              data_dgr_ch4, data_dgr_co2, data_dgr_n2, data_dgr_nocg]) 
        
        if n_subsample:
            idx = np.random.randint(0, data_all.shape[0], n_subsample)         # Subsampling index
            self.all_data = data_all.iloc[idx,:]                               # Subsample dataframe
        else:
            self.all_data = data_all
            
        if self.verbose:
            print('\n---------------- ORIGINAL DATA INFORMATION ---------------')
            print('SA:  CH4: {} | CO2: {} | N2: {} | NOCG: {}'.format(data_sa_ch4.shape, data_sa_co2.shape, data_sa_n2.shape, data_sa_nocg.shape))
            print('DGR: CH4: {} | CO2: {} | N2: {} | NOCG: {}'.format(data_dgr_ch4.shape, data_dgr_co2.shape, data_dgr_n2.shape, data_dgr_nocg.shape))
            
        if self.return_data:
            return self.all_data
        

    def process_data(self):
        '''
        Process the data to: 
        (1) shuffle, (2) truncate at large gwrt, (3) remove outliers, (4) log-transform gwrt, (5) min-max scale
        '''
        data_shuffle = self.all_data.sample(frac=1)                            # Shuffle dataset
        data_trunc = data_shuffle[data_shuffle['gwrt']<self.gwrt_cutoff]       # Truncate at gwrt threshold
        
        data_outl = data_trunc.copy()
        for col in self.y_labels:                                              # Remove outliers for outputs
            print(f"Processing column: {col}")
            data_outl = data_outl[(np.abs(zscore(data_outl[col]))<self.std_outlier)]  
        self.data_clean = data_outl

        X_data = self.data_clean.iloc[:, self.xcols]                           # Split (X,y)
        y_data = self.data_clean.iloc[:, self.ycols]
        
        sns.displot(y_data['efft']); plt.show()
        sns.displot(y_data['ymft']); plt.show()
        sns.displot(y_data['gwrt']); plt.show()
        sns.displot(y_data['injt']); plt.show()
        
        y_data_log = y_data.copy()
        y_data_log['gwrt'] = np.log10(y_data['gwrt'])                          # Log-transform gwrt
        y_data_log['injt'] = np.log10(y_data['injt'])                          # Log-transform injt
        
        self.x_scaler  = MinMaxScaler()                                        # MinMax scalar
        self.y_scaler  = MinMaxScaler() 
        self.X_dataset = self.x_scaler.fit_transform(X_data)
        self.y_dataset = self.y_scaler.fit_transform(y_data_log)
        
        # Save unscaled X_data and y_data.        
        np.save('data/X_data_unscaled.npy', X_data)
        np.save('data/y_data_unscaled.npy', y_data)              
        
        
    def plot_loss(self, title='', figsize=(6,4)):
        '''
        Plot the training performance (loss per epoch).
        '''
        plt.figure(figsize=figsize, dpi=120)  # Increased DPI for clearer saved image
        loss, val = self.metrics['training_loss'], self.metrics['validation_loss']
        epochs = len(loss)
        iterations = np.arange(epochs)
        
        # Using a clear color palette, increased line widths, and different line styles for clarity
        plt.plot(iterations, loss, '-', color='blue', lw=2.5, label='train')  
        plt.plot(iterations, val,  '-', color='darkorange', lw=2.5, label='validation') 
        
        # Title and axis labels with increased font size and a bit of padding for aesthetics
        plt.title(title + 'Training Performance', fontsize=22, pad=15)
        plt.xlabel('Epochs', fontsize=22, labelpad=10)
        plt.ylabel('Loss', fontsize=22, labelpad=10)

        # Define the ticks generation function 
        def custom_ticks(mn, mx):
            return [mn, mn + 0.25*(mx - mn), mn + 0.5*(mx - mn), mn + 0.75*(mx - mn), mx] 

        # Consistent font size for tick labels, with a grid for improved readability
        plt.xticks(custom_ticks(0, self.epochs))
        plt.yticks(custom_ticks(0, 0.15))
        # plt.xticks(iterations[::epochs//10], fontsize=12)
        plt.yticks(fontsize=22)
    
        # Legend with a slight shadow for distinction
        plt.legend(loc='upper right', fontsize=22, frameon=True, facecolor='white', edgecolor='black')
        
        # Tight layout to ensure everything fits neatly
        plt.tight_layout()
        
        # Saving the figure
        plt.savefig('figures/training_validation_loss.pdf', format='pdf')
        
        if self.return_plot:
            plt.show()

########################################################################################################################
############################################################ END #######################################################

h2 = H2Toolkit()
h2.check_torch_gpu()
h2.read_data()
h2.process_data() # or h2.load_data()
h2.train()
h2.plot_loss()
h2.evaluate()

################################################################
############################## END #############################