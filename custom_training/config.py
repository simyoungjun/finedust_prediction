import torch
class Arguments:
    
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    """
    Paths
    """
    
    block = True
    save_fig_path = './custom/result_fig/'
    save_model_path = './custom/checkpoint/best_model.pt'
    save_middle_best_model_path = './custom/checkpoint/middle_best_model.pt'
    save_last_epoch_model_path = './custom/checkpoint/last_epoch_model.pt'
    file_subname = './custom/Datasets/preprocessed_LinearInter_'
    train_path_list = ['2012-2015.csv', '2016-2019.csv', '2020-2021.csv']
    # train_path_list = ['2016-2019.csv', '2020-2021.csv']
    
    test_path = './custom/Datasets/preprocessed_LinearInter_2022.csv'

    """
            data & training setting
    """
    #Params
    MA = True
    MA_window = 5
    # districts_list = ['중구', '종로구', '성동구', '동대문구', '성북구', '용산구', '마포구', '서대문구']
    districts_list = ['중구', '강서구', '양천구', '구로구', '금천구', '관악구', '서초구', '강남구']
    # districts_list = ['중구']
    PMs = [True, True]
    days = 1
    iw = 3
    # iw = 24*days
    ow = 1
    train_stride = 1
    val_stride = 1
    test_stride = 1
    batch_size = 256
    lr = 0.0001
    epoch = 700
    min_saving_epoch = 200
    in_channel = len(districts_list)* PMs.count(True)
    train_ratio = 0.9
    max_num = 200
    segment = False
    loss_fn = 'MSE'
    # loss_fn = ['weight_L1', 'weight_MSE', MSE, L1, 'custom']
    #scheduler setting

    '''
    Model_1 network hyperparameter
    '''
    conv1d_1_kernel = 3
    if iw < conv1d_1_kernel:
        conv1d_1_kernel = iw

    conv1d_1_stride = 1
    conv1d_1_padding = 'same'
    conv1d_1_out_channels=128
    
    conv1d_2_kernel = 3
    conv1d_2_stride = 1
    conv1d_2_padding = 'same'
    conv1d_2_in_channels=conv1d_1_out_channels
    conv1d_2_out_channels=128


    lstm1_input_size=conv1d_2_out_channels
    lstm1_hidden_size=128
    lstm1_num_layers=3

    
    dense1_hidden = 64
    
    """
    Plot params
    """
    pred_graph_nrows = 3
    pred_graph_ncols = 6
    nplots = pred_graph_nrows*pred_graph_ncols
    