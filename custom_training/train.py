#%
from config import Arguments as args
from utils import *
from dataset_utils import *
from losses import *

import preprocess_finedust
import numpy as np
import torch
import os
import pandas as pd
from tqdm import tqdm
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import register_matplotlib_converters
from torch import nn, optim
from models import Conv1d_LSTM
# import wandb
# from torchsummary import summary
from copy import deepcopy

        
#%        
if __name__ == '__main__':
    
    # wandb.init(project='미세먼지 CNN-LSTM')
    # # 실행 이름 설정
    # # wandb.run.name = 'First wandb'
    # wandb.run.save()

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name())

    
    try:
        for i, filename in enumerate(args.train_path_list):
            train_path = args.file_subname + filename
            train_df_ = pd.read_csv(train_path, encoding='utf8')
            if i == 0:
                train_df = train_df_
            else:
                train_df = pd.concat([train_df, train_df_])
                
        test_df = pd.read_csv(args.test_path, encoding='utf8')
        
        
    except:
        for i, filename in enumerate(args.train_path_list):
            train_path = args.file_subname[:] + filename
            train_df_ = pd.read_csv(train_path, encoding='utf8')
            if i == 0:
                train_df = train_df_
            else:
                train_df = pd.concat([train_df, train_df_])
                
        test_df = pd.read_csv(args.test_path, encoding='utf8')
        

    # df = pd.read_csv(train_path, encoding='utf8')

    
    train_selected_df = select_feature(train_df, args.districts_list, PMs = args.PMs)
    test_selected_df = select_feature(test_df, args.districts_list, PMs = args.PMs)
    
    
    #moving average
    if args.MA == True:
        feature_cols = train_selected_df.columns
        window_size = args.MA_window
        train_selected_df[feature_cols[1:]] = train_selected_df[feature_cols[1:]].rolling(window=window_size).mean()
        #nan값 없앰
        train_selected_df = train_selected_df[24:]
        
        test_selected_df_real = test_selected_df.copy()
        test_selected_df[feature_cols[1:]] = test_selected_df[feature_cols[1:]].rolling(window=window_size).mean()
        test_selected_df = test_selected_df[24:]
        test_selected_df_real = test_selected_df_real[24:]

  #% 
    ####################
    #Params

    
    train_size = int(len(train_selected_df)*args.train_ratio//24)*24
    # val_size = int(len(selected_df)*0.2)
    train = np.array(train_selected_df[:train_size])
    val = np.array(train_selected_df[train_size::])
    test = np.array(test_selected_df)
    test_r = np.array(test_selected_df_real)
    

    train_scaler = CumtomMinMaxScaler()
    train_scaler.fit(train[:,1:], max_num=args.max_num, segment = args.segment)
    # val_scaler = CumtomMinMaxScaler()
    # val_scaler.fit(val[:,1:], max_num=args.max_num, segment = args.segment)
    # test_scaler = CumtomMinMaxScaler()
    # test_scaler.fit(test[:,1:], max_num=args.max_num, segment = args.segment)
    # scaler = None
    

    
    #데이터셋 생성
    train_dataset = CustomDataset(train, scaler=train_scaler, input_window=args.iw, output_window=args.ow, stride=args.train_stride, train_scale=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    
    val_dataset = CustomDataset(val, scaler=train_scaler, input_window=args.iw, output_window=args.ow, stride=args.val_stride, train_scale=False)
    val_loader = DataLoader(val_dataset, batch_size=len(val))

    test_dataset = CustomDataset(test, scaler=train_scaler, input_window=args.iw, output_window=args.ow, stride=args.test_stride, train_scale=False)
    test_loader = DataLoader(test_dataset, batch_size=len(test))

    test_dataset_r = CustomDataset(test_r, scaler=train_scaler, input_window=args.iw, output_window = args.ow, stride=args.test_stride, train_scale=False)
    test_loader_r = DataLoader(test_dataset_r, batch_size=len(test))

    # wandb_args = {
    # "learning_rate": args.lr,
    # "epochs": args.epoch,
    # "batch_size": args.batch_size
    # }
    # wandb.config.update(wandb_args)

    model = Conv1d_LSTM(in_channel=args.in_channel, out_channel=args.ow).to(device)

    
    # criterion = nn.MSELoss(size_average=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_date_time_y_list = train_dataset.date_time_y
    val_date_time_y_list = val_dataset.date_time_y
    test_date_time_y_list = test_dataset.date_time_y
    criterion = losses()

    
######Train
    best_model = None
    val_min_loss = 300
    progress = tqdm(range(args.epoch))
    train_losses = []
    val_losses = []
    
    for i in progress:
        model.train()
        train_batchloss = 0.0
        for (idx, X, y) in train_loader: 
            X = X.float().to(device)
            y = y.float().to(device)
            train_date_time_y = train_date_time_y_list[idx]
            weights = torch.from_numpy(train_dataset.diff_weight[idx]).to(device)
                        
            optimizer.zero_grad()
            pred = model(X)
            
            loss = criterion(pred, y, weights)
            
            # loss_max = loss_max_time_value(pred, y)
            
            loss.backward()
            optimizer.step()
            train_batchloss += loss

        progress.set_description("Train loss: {:0.6f}".format(train_batchloss.cpu().item() / len(train_loader)))

        
#Val
        with torch.no_grad():
            model.eval()
            val_batchloss = 0.0
            for (idx, X, y) in val_loader:

                X = X.float().to(device)
                y = y.float().to(device)
                val_date_time_y = val_date_time_y_list[idx]
                weights = torch.from_numpy(val_dataset.diff_weight[idx]).to(device)
                
                optimizer.zero_grad()
                pred = model(X)
                
                val_batchloss = criterion(pred, y, weights) #criterion 에는 tensortype 으로 들어가야함 (numpy X)
                
                if val_batchloss < val_min_loss:
                    val_min_loss = val_batchloss
                    best_model = deepcopy(model.state_dict())

            
            
        # wandb.log({
        #     "Train Batch total loss": train_batchloss/len(train_loader),
        #     "Val Batch total loss": val_batchloss/len(val_loader),
        # })
        train_losses.append(train_batchloss/len(train_loader))
        val_losses.append(val_batchloss/len(val_loader))
        
        if i %100 == 99:
            torch.save(best_model, args.save_model_path)
        elif i > args.min_saving_epoch and i %100 == 99:
            torch.save(best_model, args.save_middle_best_model_path)
            
    torch.save(model.state_dict(), args.save_last_epoch_model_path)
    
    # wandb.log({'loss_plot': wandb.plot.line_series(
    # xs=np.arange(args.epoch),
    # ys=[train_losses, val_losses],
    # keys=['Train Loss', 'Validation Loss'],
    # title='Train vs Validation Loss'
    # )})
    

    
    train_losses = np.array(list(map(to_numpy_cpu, train_losses)))
    val_losses = np.array(list(map(to_numpy_cpu, val_losses)))
    
    train_log_fig(np.arange(args.epoch), [train_losses, val_losses])

    

###### Model Load
    
    best_model = Conv1d_LSTM(in_channel=args.in_channel, out_channel=args.ow).to(device)
    best_model.load_state_dict(torch.load(args.save_model_path))
    best_model = best_model.to(device)
    
    # criterion = nn.MSELoss(size_average=True)
    optimizer = torch.optim.Adam(best_model.parameters(), lr=args.lr)
    #%
    test_pred = []
    test_y = []
    with torch.no_grad():
        best_model.eval()
        test_batchloss = 0.0
        for (idx, X, y) in test_loader:
            X = X.float().to(device)
            y = y.float().to(device)
            if args.MA == True:
                idx_r, X_r, y_r = test_dataset_r[idx]
                y_r = torch.from_numpy(y_r).float().to(device)
            test_date_time_y = test_date_time_y_list[idx]
            weights = torch.from_numpy(test_dataset.diff_weight[idx]).to(device)
            
            optimizer.zero_grad()
            pred = best_model(X)
            
            # if scaler != None:
            #     test_pred_unscaled = scaler.inverse_transform_for_test(pred.cpu().numpy(), i=0)
            #     test_y_unscaled = scaler.inverse_transform_for_test(y.cpu().numpy(), i=0)

            # loss = criterion(pred, y, weights)
            loss = criterion(pred, y, weights)
            

            # loss.backward()
            # optimizer.step()
            test_batchloss += loss
            
    
        # test_pred = test_scaler.inverse_transform_for_GT(pred.cpu().numpy(), i=0)
        # test_y = test_scaler.inverse_transform_for_GT(y_r.cpu().numpy(), i=0)
        test_pred = train_scaler.inverse_transform_for_GT(pred.cpu().numpy(), i=0)
        test_y = train_scaler.inverse_transform_for_GT(y.cpu().numpy(), i=0)
        test_y_r = train_scaler.inverse_transform_for_GT(y_r.cpu().numpy(), i=0)

        test_pred = merge_results(test_pred)
        test_y = merge_results(test_y)
        test_y_r = merge_results(test_y_r)
        
        test_pred_max = np.max(test_pred, axis=1)
        test_pred_argmax = np.argmax(test_pred, axis=1)
        test_y_max = np.max(test_y, axis=1)
        test_y_argmax = np.argmax(test_y, axis=1)  
            
    print("Test Batch total loss : ", test_batchloss/len(test_loader))
    # wandb.log({
    #     "Test Batch total loss": test_batchloss/len(test_loader),
    # })
#%
    #plot test 결과

    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(30, 6))
    
    total_result_plt = scatter_plot_result(test_pred, test_y, 'Total result', axes=axes, sub_idx = 0)
    
    kde_plot(test_y, test_pred, [0,250], fig, axes, sub_idx=1)
    
    max_value_result_plt = scatter_plot_result(test_pred_max, test_y_max, 'Max values result', axes=axes, sub_idx = 2)
    
    kde_plot(test_y_max, test_pred_max, [0,250], fig, axes, sub_idx=3)
    
    max_time_heatmap_plt = max_time_heatmap(test_pred_argmax, test_y_argmax, 'Argmax values result', xy_lim=[-1, 30], axes=axes, sub_idx = 4)

    plt.tight_layout()
    plt.show()
    fig.savefig(args.save_fig_path+"results.png")
    
    # wandb.log({
    # "Total result": wandb.Image(total_result_plt),
    # "Max values result": wandb.Image(max_value_result_plt),
    # "Argmax values result": wandb.Image(max_time_heatmap_plt),
    # })
    test_date_time_y = merge_results(test_date_time_y)

    fig1, axes1 = plt.subplots(nrows=args.pred_graph_nrows, ncols=args.pred_graph_ncols, figsize=(24, 16))
    
    _ = 0
    for i in range(len(test_y)):
        if i%20 == 0:
            graph_plt = plot_graph(test_date_time_y[i], test_pred[i], test_y[i], i, axes=axes1, sub_idx = _, y_r = test_y_r[i])
            _ += 1
        if _ == args.nplots:
            break 
        
    plt.tight_layout()
    plt.show()
    fig1.savefig(args.save_fig_path+"graph_pred.png")
    print('END')