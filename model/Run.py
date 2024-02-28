import os
import sys
import json
file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(file_dir)
sys.path.append(file_dir)

import torch
import numpy as np
import torch.nn as nn
from datetime import datetime
from model.AGCRN import AGCRN as Network
from model.BasicTrainer import Trainer
from lib.TrainInits import init_seed
from lib.dataloader import get_dataloader
from lib.TrainInits import print_model_parameters


#*************************************************************************#
Mode = 'Train'
DEBUG = 'True'
DATASET = 'SAMPLES'      #PEMSD4 or PEMSD8  or SAMPLES
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL = 'AGCRN'

#get configuration
config_file = './config_{}.json'.format(MODEL)
#print('Read configuration file: %s' % (config_file))
with open(config_file, 'r') as f:
    config = json.loads(f.read())

from lib.metrics import MAE_torch
def masked_mae_loss(scaler, mask_value):
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        mae = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
        return mae
    return loss

class args_AGCRN():
    def __init__(self,config):
        #parser
        self.dataset = DATASET
        self.mode = Mode
        self.device = DEVICE
        self.debug = DEBUG
        self.model = MODEL
        self.cuda = True
        #data
        self.val_ratio = config['data']['val_ratio']
        self.test_ratio = config['data']['test_ratio']
        self.lag = config['data']['lag']
        self.horizon = config['data']['horizon']
        self.num_nodes = config['data']['num_nodes']
        self.tod = config['data']['tod']
        self.normalizer = config['data']['normalizer']
        self.column_wise = config['data']['column_wise']
        self.default_graph = config['data']['default_graph']
        #model
        self.input_dim = config['model']['input_dim']
        self.output_dim = config['model']['output_dim']
        self.embed_dim = config['model']['embed_dim']
        self.rnn_units = config['model']['rnn_units']
        self.num_layers = config['model']['num_layers']
        self.cheb_k = config['model']['cheb_order']
        #train
        self.loss_func = config['train']['loss_func']
        self.seed = config['train']['seed']
        self.batch_size = config['train']['batch_size']
        self.epochs = config['train']['epochs']
        self.lr_init = config['train']['lr_init']
        self.lr_decay = config['train']['lr_decay']
        self.lr_decay_rate = config['train']['lr_decay_rate']
        self.lr_decay_step = config['train']['lr_decay_step']
        self.early_stop = config['train']['early_stop']
        self.early_stop_patience = config['train']['early_stop_patience']
        self.grad_norm = config['train']['grad_norm']
        self.max_grad_norm = config['train']['max_grad_norm']
        self.teacher_forcing = False
        self.real_value = config['train']['real_value']
        #test
        self.mae_thresh = config['test']['mae_thresh']
        self.mape_thresh = config['test']['mape_thresh']
        #log
        self.log_dir = './'
        self.log_step = config['log']['log_step']
        self.plot = config['log']['plot']
args = args_AGCRN(config)

init_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.set_device(int(args.device[5]))
else:
    args.device = 'cpu'

#init model
model = Network(args)
model = model.to(args.device)
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
    else:
        nn.init.uniform_(p)
print_model_parameters(model, only_num=False)

#load dataset
train_loader, val_loader, test_loader, scaler = get_dataloader(args,
                                                               normalizer=args.normalizer,
                                                               tod=args.tod, dow=False,
                                                               weather=False, single=False)

#init loss function, optimizer
if args.loss_func == 'mask_mae':
    loss = masked_mae_loss(scaler, mask_value=0.0)
elif args.loss_func == 'mae':
    loss = torch.nn.L1Loss().to(args.device)
elif args.loss_func == 'mse':
    loss = torch.nn.MSELoss().to(args.device)
else:
    raise ValueError

optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_init, eps=1.0e-8,
                             weight_decay=0, amsgrad=False)
#learning rate decay
lr_scheduler = None
if args.lr_decay:
    print('Applying learning rate decay.')
    lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(','))]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                        milestones=lr_decay_steps,
                                                        gamma=args.lr_decay_rate)
    #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=64)

#config log path
current_time = datetime.now().strftime('%Y%m%d%H%M%S')
current_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(current_dir,'experiments', args.dataset, current_time)
args.log_dir = log_dir
args.log_dir = os.path.join('/content/gdrive/MyDrive/Models/AGCRN','experiments',current_time)

#start training
trainer = Trainer(model, loss, optimizer, train_loader, val_loader, test_loader, scaler,
                  args, lr_scheduler=lr_scheduler)
if args.mode == 'Train':
    trainer.train()
elif args.mode == 'Test':
    model.load_state_dict(torch.load('../pre-trained/{}.pth'.format(args.dataset)))
    print("Load saved model")
    trainer.test(model, trainer.args, test_loader, scaler, trainer.logger)
else:
    raise ValueError
