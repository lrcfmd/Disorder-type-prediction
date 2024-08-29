import numpy as np
import pandas as pd
import collections
from collections import OrderedDict
import pytorch_lightning as L
import os
import re
import json
import tqdm

from sklearn.metrics import balanced_accuracy_score, accuracy_score, roc_auc_score, f1_score, average_precision_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, matthews_corrcoef, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split

from pytorch_lightning.loggers.csv_logs import CSVLogger
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
# from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer
from torchmetrics.functional import mean_squared_error, mean_absolute_error

from pymatgen.core.composition import Composition
from crabnet.kingcrab import CrabNet

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CyclicLR, CosineAnnealingLR, StepLR

from crabnet.utils.utils import (Lamb, Lookahead, RobustL1, CrossEntropyLoss,
                         EDMDataset, get_edm, Scaler, DummyScaler, count_parameters)
from crabnet.utils.get_compute_device import get_compute_device
# from crabnet.utils.composition import _element_composition, get_sym_dict, parse_formula, CompositionError
#from utils.optim import SWA

data_type_np = np.float32
data_type_torch = torch.float32

import wandb


class CrabNetDataModule(L.LightningDataModule):
    def __init__(self, train_file: str , 
                 val_file: str, 
                 test_file: str,
                 n_elements ='infer', 
                 classification = False,
                 elem_prop='mat2vec',
                 batch_size = 2**10,
                 scale = True,
                 pin_memory = True):
        super().__init__()
        self.train_path = train_file
        self.val_path = val_file
        self.test_path = test_file
        self.batch_size = batch_size
        self.n_elements=n_elements
        self.pin_memory = pin_memory
        self.scale = scale
        self.classification = classification
        self.elem_prop=elem_prop

    def prepare_data(self):
        ### loading and encoding trianing data
        if(re.search('.json', self.train_path )):
            self.data_train=pd.read_json(self.train_path)
        elif(re.search('.csv', self.train_path)):
            self.data_train=pd.read_csv(self.train_path)

        self.train_main_data = list(get_edm(self.data_train, elem_prop=self.elem_prop,
                                      n_elements=self.n_elements,
                                      inference=False,
                                      verbose=True,
                                      drop_unary=False,
                                      scale=self.scale))
        
        self.train_len_data = len(self.train_main_data[0])
        self.train_n_elements = self.train_main_data[0].shape[1]//2

        print(f'loading data with up to {self.train_n_elements:0.0f} '
              f'elements in the formula for training')
        
        ### loading and encoding validation data
        if(re.search('.json', self.val_path )):
            self.data_val=pd.read_json(self.val_path)
        elif(re.search('.csv', self.val_path)):
            self.data_val=pd.read_csv(self.val_path)
        
        self.val_main_data = list(get_edm(self.data_val, elem_prop=self.elem_prop,
                                      n_elements=self.n_elements,
                                      inference=True,
                                      verbose=True,
                                      drop_unary=False,
                                      scale=self.scale))
        
        self.val_len_data = len(self.val_main_data[0])
        self.val_n_elements = self.val_main_data[0].shape[1]//2

        print(f'loading data with up to {self.val_n_elements:0.0f} '
              f'elements in the formula for validation')
        
        ### loading and encoding testing data
        if(re.search('.json', self.test_path )):
            self.data_test=pd.read_json(self.test_path)
        elif(re.search('.csv', self.test_path)):
            self.data_test=pd.read_csv(self.test_path)
        
        self.test_main_data = list(get_edm(self.data_test, elem_prop=self.elem_prop,
                                      n_elements=self.n_elements,
                                      inference=True,
                                      verbose=True,
                                      drop_unary=False,
                                      scale=self.scale))
        
        self.test_len_data = len(self.test_main_data[0])
        self.test_n_elements = self.test_main_data[0].shape[1]//2

        print(f'loading data with up to {self.test_n_elements:0.0f} '
              f'elements in the formula for testing')

        self.train_dataset = EDMDataset(self.train_main_data, self.train_n_elements)
        self.val_dataset = EDMDataset(self.val_main_data, self.val_n_elements)
        self.test_dataset = EDMDataset(self.test_main_data, self.test_n_elements)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          pin_memory=self.pin_memory, shuffle=True)
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                        pin_memory=self.pin_memory, shuffle=False)
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.test_len_data,
                        pin_memory=self.pin_memory, shuffle=False)
    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.test_len_data,
                        pin_memory=self.pin_memory, shuffle=False)


class CrabNetLightning(L.LightningModule):
    def __init__(self, **config):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters()

        self.model = CrabNet(out_dims=config['out_dims'],
                             d_model=config['d_model'],
                             N=config['N'],
                             heads=config['heads'])
        print('\nModel architecture: out_dims, d_model, N, heads')
        print(f'{self.model.out_dims}, {self.model.d_model}, '
                  f'{self.model.N}, {self.model.heads}')
        print(f'Model size: {count_parameters(self.model)} parameters\n')

        ### here we define some important parameters
        self.fudge=config['fudge']
        self.batch_size=config['batch_size']
        self.classification = config['classification']
        self.base_lr=config['base_lr']
        self.max_lr=config['max_lr']
        
        self.criterion = CrossEntropyLoss
        
        if(re.search('.json', config['train_path'] )):
            train_data=pd.read_json(config['train_path'])
        elif(re.search('.csv', config['train_path'])):
            train_data=pd.read_csv(config['train_path'])
        
        y=train_data['target'].values
        self.step_size = len(y)

    def forward(self, src, frac):
        out=self.model(src, frac)
        return out

    def configure_optimizers(self):
        base_optim = Lamb(params=self.model.parameters(),lr=0.001)
        optimizer = Lookahead(base_optimizer=base_optim)
        lr_scheduler = CyclicLR(optimizer,
                                base_lr=self.base_lr,
                                max_lr=self.max_lr,
                                cycle_momentum=False,
                                step_size_up=self.step_size)
        # lr_scheduler=StepLR(optimizer,
        #                     step_size=3,
        #                     gamma=0.5)
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        X, y, formula = batch
        src, frac = X.squeeze(-1).chunk(2, dim=1)
        frac = frac * (1 + (torch.randn_like(frac))*self.fudge)
        frac = torch.clamp(frac, 0, 1)
        frac[src == 0] = 0
        frac = frac / frac.sum(dim=1).unsqueeze(1).repeat(1, frac.shape[-1])
        
        prediction = self(src, frac)
        loss = self.criterion(prediction,y)
        
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        
        prediction = torch.nn.functional.softmax(prediction,dim=1)
        y_pred = torch.argmax(prediction,dim=1)
        y=y.detach().cpu().numpy()
        y_pred=y_pred.detach().cpu().numpy()
        acc=balanced_accuracy_score(y,y_pred)
        f1=f1_score(y,y_pred,average='macro')
        mc=matthews_corrcoef(y,y_pred)
            
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        self.log("train_f1", f1, on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)
        self.log("train_mc", mc, on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)
        
        return loss
     
    def validation_step(self, batch, batch_idx):
        X, y, formula = batch
        src, frac = X.squeeze(-1).chunk(2, dim=1)
        frac = frac * (1 + (torch.randn_like(frac))*self.fudge)
        frac = torch.clamp(frac, 0, 1)
        frac[src == 0] = 0
        frac = frac / frac.sum(dim=1).unsqueeze(1).repeat(1, frac.shape[-1])
        
        prediction = self(src, frac)
        val_loss = self.criterion(prediction,y)
        
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        
        prediction = torch.nn.functional.softmax(prediction,dim=1)
        y_pred = torch.argmax(prediction,dim=1)
        y=y.detach().cpu().numpy()
        y_pred=y_pred.detach().cpu().numpy()
        acc=balanced_accuracy_score(y,y_pred)
        f1=f1_score(y,y_pred,average='macro')
        mc=matthews_corrcoef(y,y_pred)
            
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        self.log("val_f1", f1, on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)
        self.log("val_mc", mc, on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)
        return val_loss
     
    def test_step(self, batch, batch_idx):
        X, y, formula = batch
        src, frac = X.squeeze(-1).chunk(2, dim=1)
        frac = frac * (1 + (torch.randn_like(frac))*self.fudge)
        frac = torch.clamp(frac, 0, 1)
        frac[src == 0] = 0
        frac = frac / frac.sum(dim=1).unsqueeze(1).repeat(1, frac.shape[-1])
        
        prediction = self(src, frac)
        
        prediction = torch.nn.functional.softmax(prediction,dim=1)
        y_pred = torch.argmax(prediction,dim=1)
        y=y.detach().cpu().numpy()
        y_pred=y_pred.detach().cpu().numpy()
        acc=balanced_accuracy_score(y,y_pred)
        f1=f1_score(y,y_pred,average='macro')
        mc=matthews_corrcoef(y,y_pred)
            
        self.log("test_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        self.log("test_f1", f1, on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)
        self.log("test_mc", mc, on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)
        return
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        X, y, formula = batch
        src, frac = X.squeeze(-1).chunk(2, dim=1)
        frac = frac * (1 + (torch.randn_like(frac))*self.fudge)
        frac = torch.clamp(frac, 0, 1)
        frac[src == 0] = 0
        frac = frac / frac.sum(dim=1).unsqueeze(1).repeat(1, frac.shape[-1])
        
        prediction = self(src, frac)
        prediction = torch.nn.functional.softmax(prediction,dim=1)
        y_pred = torch.argmax(prediction,dim=1)
        y=y.detach().cpu().numpy()
        y_pred=y_pred.detach().cpu().numpy()
        acc=balanced_accuracy_score(y,y_pred)
        f1=f1_score(y,y_pred,average='macro')
        mc=matthews_corrcoef(y,y_pred)
            
        self.log("predict_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        self.log("predict_f1", f1, on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)
        self.log("predict_mc", mc, on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)
        
        return formula, y_pred, prediction

def main(**config):
    model = CrabNetLightning(**config)
    wandb_logger = WandbLogger(project="Crabnet-global-disorder-type", config=config, log_model="all")
    trainer = Trainer(max_epochs=100,accelerator='gpu', devices=1, logger=wandb_logger,
                      callbacks=[StochasticWeightAveraging(swa_epoch_start=config['swa_epoch_start'],swa_lrs=config['swa_lrs']),
                                EarlyStopping(monitor='val_loss', mode='min', patience=config['patience']), ModelCheckpoint(monitor='val_acc', mode="max", 
                                dirpath='crabnet_models/crabnet_trained_models/', filename='disorder-{epoch:02d}-{val_acc:.2f}')])
    disorder_data = CrabNetDataModule(config['train_path'],
                                   config['val_path'],
                                   config['test_path'],
                                   classification = config['classification'])
    trainer.fit(model, datamodule=disorder_data)
    trainer.test(ckpt_path='best',datamodule=disorder_data)
    for x in disorder_data.predict_dataloader():
        _, y_true, _ = x
    formula, y_pred, prediction=trainer.predict(ckpt_path='best', datamodule=disorder_data)[0]
    
    metrics={}
    metrics['acc']=balanced_accuracy_score(y_true,y_pred)
    metrics['f1']=f1_score(y_true,y_pred,average='macro')
    metrics['precision']=average_precision_score(y_true,y_pred,average='macro')
    metrics['recall']=recall_score(y_true,y_pred,average='macro')
    metrics['mc']=matthews_corrcoef(y_true,y_pred)
    
    
    pred_matrix={}
    pred_matrix['y_true']=y_true
    pred_matrix['y_score']=prediction.detach().numpy()
    pred_matrix['y_true']=y_pred
   
    wandb.log(metrics)
    wandb.log(pred_matrix)


    return
    

if __name__=='__main__':
    wandb.init(project="Crabnet-global-disorder-type")
    wandb.login(key='b11d318e434d456c201ef1d3c86a3c1ce31b98d7')

    with open('crabnet/crabnet_config.json','r') as f:
        config=json.load(f)

    L.seed_everything(config['random_seed'])
    main(**config)

    wandb.finish()
    # print('Start sweeping with different parameters for RF...')

    # wandb.login(key='b11d318e434d456c201ef1d3c86a3c1ce31b98d7')

    # sweep_config = {
    # 'method': 'random',
    # 'parameters': {'n_estimators': {'values': [50, 100, 150, 200]},
    #                'class_weight': {'values':['balanced', 'balanced_subsample']},
    #                'criterion': {'values': ['gini', 'entropy', 'log_loss']}
    # }
    # }

    # sweep_id = wandb.sweep(sweep=sweep_config, project="RF-disorder-prediction-global-disorder")

    # wandb.agent(sweep_id, function=main, count=10)

    # wandb.finish()
