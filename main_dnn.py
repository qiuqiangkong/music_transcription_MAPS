"""
Summary:  Piano automatic music transcription (AMT) on MAPS dataset. 
Author:   Qiuqiang Kong
Created:  2017.12.11
Modified: 
"""
from __future__ import print_function
import os
import numpy as np
import csv
import time
import pickle
import cPickle
import h5py
import argparse
import matplotlib.pyplot as plt

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import config as cfg
import prepare_data as pp_data
from data_generator import DataGenerator

feat_type = 'logmel'

def uniform_weights(m):
    classname = m.__class__.__name__    
    if classname.find('Linear') != -1:
        scale = 0.1
        m.weight.data = torch.nn.init.uniform(m.weight.data, -scale, scale)
        m.bias.data.fill_(0.)

def glorot_uniform_weights(m):
    classname = m.__class__.__name__    
    if classname.find('Linear') != -1:
        # w = torch.nn.init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
        w = torch.nn.init.xavier_uniform(m.weight.data)
        m.weight.data = w
        m.bias.data.fill_(0.)    

# Evaluate on batch. 
def eval(model, gen, xs, ys, cuda):
    model.eval()
    pred_all = []
    y_all = []
    for (batch_x, batch_y) in gen.generate(xs=xs, ys=ys):
        batch_x = torch.Tensor(batch_x)
        batch_x = Variable(batch_x, volatile=True)
        if cuda:
            batch_x = batch_x.cuda()
        pred = model(batch_x)
        pred = pred.data.cpu().numpy()
        pred_all.append(pred)
        y_all.append(batch_y)
        
    pred_all = np.concatenate(pred_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    (tp, fn, fp, tn) = pp_data.tp_fn_fp_tn(pred_all, y_all, thres=0.5, average='micro')
    (prec, recall, fvalue) = pp_data.prec_recall_fvalue(pred_all, y_all, thres=0.5, average='micro')
    
    # Debug. 
    if False:
        print("tp, fn, fp, tn: %d, %d, %d, %d" % (tp, fn, fp, tn))
        
    print("prec: %f, recall: %f, fvalue: %f" % (prec, recall, fvalue))

class Net(nn.Module):
    def __init__(self, n_concat, n_freq, n_out):
        super(Net, self).__init__()
        n_in = n_concat * n_freq
        n_hid = 500
        
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_hid)
        self.fc3 = nn.Linear(n_hid, n_hid)
        self.fc4 = nn.Linear(n_hid, n_out)
        
    def forward(self, x):
        drop_p = 0.2
        x1 = x.view(len(x), -1)
        x2 = F.dropout(F.relu(self.fc1(x1)), p=drop_p, training=self.training)
        x3 = F.dropout(F.relu(self.fc2(x2)), p=drop_p, training=self.training)
        x4 = F.dropout(F.relu(self.fc3(x3)), p=drop_p, training=self.training)
        x5 = F.sigmoid(self.fc4(x4))
        return x5
        

def train(args):
    cuda = args.use_cuda and torch.cuda.is_available()
    workspace = args.workspace
    feat_type = args.feat_type
    lr = args.lr
    resume_model_path = args.resume_model_path
    script_na = args.script_na
    print("cuda:", cuda)

    # Load data. 
    t1 = time.time()
    tr_packed_feat_path = os.path.join(workspace, "packed_features", feat_type, "train.p")
    te_packed_feat_path = os.path.join(workspace, "packed_features", feat_type, "test.p")
    [tr_x_list, tr_y_list, tr_na_list] = cPickle.load(open(tr_packed_feat_path, 'rb'))
    [te_x_list, te_y_list, te_na_list] = cPickle.load(open(te_packed_feat_path, 'rb'))
    print("Loading packed feature time: %s s" % (time.time() - t1,))
        
    # Scale. 
    if True:
        scale_path = os.path.join(workspace, "scalers", feat_type, "scaler.p")
        scaler = pickle.load(open(scale_path, 'rb'))
        tr_x_list = pp_data.scale_on_x_list(tr_x_list, scaler)
        te_x_list = pp_data.scale_on_x_list(te_x_list, scaler)

    # Debug. 
    if False:
        fig, axs = plt.subplots(2,1, sharex=True)
        axs[0].matshow(tr_x_list[0].T, origin='lower', aspect='auto')
        axs[1].matshow(tr_y_list[0].T, origin='lower', aspect='auto')
        plt.show()
        pause
    
    # Data to 3d. 
    n_concat = 3
    n_hop = 1
    (tr_x, tr_y) = pp_data.data_to_3d(tr_x_list, tr_y_list, n_concat, n_hop)
    (te_x, te_y) = pp_data.data_to_3d(te_x_list, te_y_list, n_concat, n_hop)
    n_freq = tr_x.shape[-1]
    n_out = tr_y.shape[-1]
    print(tr_x.shape, tr_y.shape)
    
    # Model. 
    model = Net(n_concat, n_freq, n_out)
    
    if os.path.isfile(resume_model_path):
        # Load weights. 
        print("Loading checkpoint '%s'" % resume_model_path)
        checkpoint = torch.load(resume_model_path)
        model.load_state_dict(checkpoint['state_dict'])
        iter = checkpoint['iter']
    else:
        # Randomly init weights. 
        print("Train from random initialization. ")
        model.apply(glorot_uniform_weights)
        iter = 0
    
    # Move model to GPU. 
    if cuda:
        model.cuda()
    
    # Optimizer. 
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    
    # Data Generator
    batch_size = 500
    tr_gen = DataGenerator(batch_size=batch_size, type='train')
    eval_tr_gen = DataGenerator(batch_size=batch_size, type='test', te_max_iter=20)
    eval_te_gen = DataGenerator(batch_size=batch_size, type='test')
    
    iters_per_epoch = len(tr_x) / batch_size
    print("Iters_per_epoch: %d" % iters_per_epoch)
    
    # Train. 
    eps = 1e-8
    tr_time = 0
    for (batch_x, batch_y) in tr_gen.generate(xs=[tr_x], ys=[tr_y]):
        if iter % (1000) == 0:
            print("\n--- Evaluation of training set (subset), iteration: %d ---" % iter)
            eval(model, eval_tr_gen, [tr_x], [tr_y], cuda)
            print("--- Evaluation of testing set, iteration: %d ---" % iter)
            eval(model, eval_te_gen, [te_x], [te_y], cuda)
            print("-----------------------------------------------\n")
        
        # Move data to GPU. 
        t1 = time.time()
        batch_x = torch.Tensor(batch_x)
        batch_y = torch.Tensor(batch_y)
        batch_x = Variable(batch_x)
        batch_y = Variable(batch_y)
        if cuda:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
        optimizer.zero_grad()
        model.train()
        output = model(batch_x)
        output = torch.clamp(output, eps, 1. - eps)
        
        loss = F.binary_cross_entropy(output, batch_y)
        loss.backward()
        optimizer.step()
        
        if iter % 200 == 0:
            print("Iter: %d loss: %f" % (iter, loss))
        
        iter += 1
        
        # Save model. 
        if iter % 1000 == 0:
            save_out_dict = {'iter': iter, 
                             'state_dict': model.state_dict(), 
                             'optimizer': optimizer.state_dict(), }
            save_out_path = os.path.join(workspace, "models", script_na, feat_type, "md_%diters.tar" % iter)
            pp_data.create_folder(os.path.dirname(save_out_path))
            torch.save(save_out_dict, save_out_path)
            print("Save model to %s" % save_out_path)
            
        # Stop training. 
        if iter == 10001:
            break

def inference(args):
    cuda = args.use_cuda and torch.cuda.is_available()
    workspace = args.workspace
    model_name = args.model_name
    feat_type = args.feat_type
    script_na = args.script_na

    # Load data. 
    te_packed_feat_path = os.path.join(workspace, "packed_features", feat_type, "test.p")
    [te_x_list, te_y_list, te_na_list] = cPickle.load(open(te_packed_feat_path, 'rb'))
        
    # Scale. 
    if True:
        scale_path = os.path.join(workspace, "scalers", feat_type, "scaler.p")
        scaler = pickle.load(open(scale_path, 'rb'))
        te_x_list = pp_data.scale_on_x_list(te_x_list, scaler)
        
    # Construct model topology. 
    n_concat = 3
    te_n_hop = 1
    n_freq = te_x_list[0].shape[-1]
    n_out = te_y_list[0].shape[-1]
    model = Net(n_concat, n_freq, n_out)
    
    # Init the weights of model using trained weights. 
    model_path = os.path.join(workspace, "models", script_na, feat_type, model_name)
    if os.path.isfile(model_path):
        print("Loading checkpoint '%s'" % model_path)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        raise Exception("Model path %s does not exist!" % model_path)
        
    # Move model to GPU. 
    if cuda:
        model.cuda()
        
    # Directory to write out transcript midi files. 
    out_midi_dir = os.path.join(workspace, "out_midis", pp_data.get_filename(__file__), feat_type)
    pp_data.create_folder(out_midi_dir)
        
    # Data to 3d. 
    n_half = (n_concat - 1) / 2
    for i1 in xrange(len(te_x_list)):
        x = te_x_list[i1]   # (n_time, n_freq)
        y = te_y_list[i1]   # (n_time, n_out)
        bare_na = os.path.splitext(te_na_list[i1])[0]
        (n_time, n_freq) = x.shape
        
        zero_pad = np.zeros((n_half, n_freq))
        x = np.concatenate((zero_pad, x, zero_pad), axis=0)
        x3d = pp_data.mat_2d_to_3d(x, n_concat, te_n_hop)     # (n_time, n_concat, n_freq)
        
        # Move data to GPU. 
        x3d = torch.Tensor(x3d)
        x3d = Variable(x3d)
        if cuda:
            x3d = x3d.cuda()
        
        # Inference. 
        model.eval()
        pred = model(x3d)   # (n_time, n_out)
        
        # Convert data type to numpy. 
        pred = pred.data.cpu().numpy()
        
        # Threshold and write out predicted piano roll to midi file. 
        mid_roll = pp_data.prob_to_midi_roll(pred, 0.5)
        out_path = os.path.join(out_midi_dir, "%s.mid" % bare_na)
        print("Write out to: %s" % out_path)
        pp_data.write_midi_roll_to_midi(mid_roll, out_path)
        
        # Debug. 
        if True:
            fig, axs = plt.subplots(3,1, sharex=True)
            axs[0].matshow(y.T, origin='lower', aspect='auto')
            axs[1].matshow(pred.T, origin='lower', aspect='auto')
            binary_pred = (np.sign(pred - 0.5) + 1) / 2
            axs[2].matshow(binary_pred.T, origin='lower', aspect='auto')
            axs[0].set_title("Ground truth")
            axs[1].set_title("Output")
            axs[2].set_title("Threshold output")
            plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--use_cuda', action='store_true', default=True)
    parser_train.add_argument('--workspace', type=str)
    parser_train.add_argument('--feat_type', type=str, choices=['logmel'])
    parser_train.add_argument('--lr', type=float, default=1e-3)
    parser_train.add_argument('--resume_model_path', type=str, default="")
                    
    parser_inference = subparsers.add_parser('inference')
    parser_inference.add_argument('--use_cuda', action='store_true', default=True)
    parser_inference.add_argument('--workspace', type=str)
    parser_inference.add_argument('--model_name', type=str)
    parser_inference.add_argument('--feat_type', type=str, choices=['logmel'])
    
    args = parser.parse_args()

    if args.mode == "train":
        args.script_na = pp_data.get_filename(__file__)
        train(args)
    elif args.mode == "inference":
        args.script_na = pp_data.get_filename(__file__)
        inference(args)
    else:
        raise Exception("Incorrect argument!")