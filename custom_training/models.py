import torch.nn as nn
import module as mm
import torch
from config import Arguments as args

class Conv1d_LSTM_Resnet(nn.Module):
    def __init__(self, in_channel=args.in_channel, out_channel=args.ow):
        super(Conv1d_LSTM_Resnet, self).__init__()
        self.conv1d_1 = mm.Conv1d(in_channels=in_channel,
                                out_channels=args.conv1d_1_out_channels,
                                kernel_size=args.conv1d_1_kernel,
                                stride=args.conv1d_1_stride,
                                padding= args.conv1d_1_padding,
                                activation_fn = nn.ReLU())
        
        self.conv1d_2 = mm.Conv1d(in_channels=args.conv1d_1_out_channels,
                                out_channels=args.conv1d_2_out_channels,
                                kernel_size=args.conv1d_2_kernel,
                                stride=args.conv1d_2_stride,
                                padding=args.conv1d_2_padding,
                                activation_fn = nn.ReLU())
        
        self.lstm_1 = nn.LSTM(input_size=args.lstm1_input_size,
                            hidden_size=args.lstm1_hidden_size,
                            num_layers=args.lstm1_num_layers,
                            bias=True,
                            bidirectional=False,
                            batch_first=True)
        
        self.Conv1dResBlock_1 = mm.Conv1dResBlock(in_channels=in_channel,
                                out_channels=args.in_channel,
                                kernel_size=args.conv1d_1_kernel,
                                stride=args.conv1d_1_stride,
                                padding= args.conv1d_1_padding,
                                activation_fn = nn.ReLU())
        
        self.Conv1dResBlock_2 = mm.Conv1dResBlock(in_channels=args.conv1d_1_out_channels,
                                out_channels=args.conv1d_1_out_channels,
                                kernel_size=args.conv1d_2_kernel,
                                stride=args.conv1d_2_stride,
                                padding=args.conv1d_2_padding,
                                activation_fn = nn.ReLU())
        
        self.dropout = nn.Dropout(0.5)

        # self.dense1 = nn.Linear(args.lstm1_hidden_size, args.dense1_hidden)
        self.dense2 = nn.Linear(args.lstm1_hidden_size, out_channel)
        
        self.relu = nn.ReLU()
        # self.batch_norm = nn.BatchNorm1d(args.lstm1_hidden_size)

    def forward(self, x):
	# Raw x shape : (B, S, F) => (B, 10, 3)
        
        # Shape : (B, F, S) => (B, 3, 10)
        # x = x.transpose(1, 2)
        # Shape : (B, F, S) == (B, C, S) // C = channel => (B, 16, 10)
        # x = self.Conv1dResBlock_1(x)
        x = self.conv1d_1(x)
        # Shape : (B, C, S) => (B, 32, 10)
        # x = self.conv1d_2(x)
        x = self.Conv1dResBlock_2(x)
        # Shape : (B, S, C) == (B, S, F) => (B, 10, 32)
        # x = x.transpose(1, 2)
        
        self.lstm_1.flatten_parameters()
        # self.lstm_2.flatten_parameters()
        
        # Shape : (B, S, H) // H = hidden_size => (B, 10, 50)
        x, (hidden, _) = self.lstm_1(x)
        
        # x, (hidden, _) = self.lstm_2(x)
        
        # x = self.batch_norm(x)
        # Shape : (B, H) // -1 means the last sequence => (B, 50)
        x = hidden[-1]
        # x = x[:, -1, :]
        # Shape : (B, H) => (B, 50)
        # x = self.dropout(x)
        
        # Shape : (B, 32)
        # x = self.dense1(x)
        # Shape : (B, O) // O = output => (B, 1)
        x = self.dense2(x)

        return x
    
    def save(self, fn):
        ''' Save the model to a file.
        
        Args:
            fn (str): filename.
        '''
        torch.save(self.state_dict(), fn)
    
    def load(self, fn):
        ''' Load the model from a file.
        
        Args:
            fn (str): filename.
        '''
        state_dict = torch.load(fn)
        self.load_state_dict(state_dict)
        
        
class Conv1d_LSTM(nn.Module):
    def __init__(self, in_channel=args.in_channel, out_channel=args.ow):
        super(Conv1d_LSTM, self).__init__()
        self.conv1d_1 = nn.Conv1d(in_channels=in_channel,
                                out_channels=args.conv1d_1_out_channels,
                                kernel_size=args.conv1d_1_kernel,
                                stride=args.conv1d_1_stride,
                                padding= args.conv1d_1_padding)
        self.conv1d_2 = nn.Conv1d(in_channels=args.conv1d_2_in_channels,
                                out_channels=args.conv1d_2_out_channels,
                                kernel_size=args.conv1d_2_kernel,
                                stride=args.conv1d_2_stride,
                                padding=args.conv1d_2_padding)
        
        self.lstm = nn.LSTM(input_size=args.lstm1_input_size,
                            hidden_size=args.lstm1_hidden_size,
                            num_layers=args.lstm1_num_layers,
                            bias=True,
                            bidirectional=False,
                            batch_first=True)
        
        self.dropout = nn.Dropout(0.5)

        self.dense1 = nn.Linear(args.lstm1_hidden_size, args.dense1_hidden)
        self.dense2 = nn.Linear(args.lstm1_hidden_size, out_channel)

    def forward(self, x):
	# Raw x shape : (B, S, F) => (B, 10, 3)
        
        # Shape : (B, F, S) => (B, 3, 10)
        x = x.transpose(1, 2)
        # Shape : (B, F, S) == (B, C, S) // C = channel => (B, 16, 10)
        x = self.conv1d_1(x)
        # Shape : (B, C, S) => (B, 32, 10)
        x = self.conv1d_2(x)
        # Shape : (B, S, C) == (B, S, F) => (B, 10, 32)
        x = x.transpose(1, 2)
        
        self.lstm.flatten_parameters()
        # Shape : (B, S, H) // H = hidden_size => (B, 10, 50)
        _, (hidden, _) = self.lstm(x)
        # Shape : (B, H) // -1 means the last sequence => (B, 50)
        x = hidden[-1]
        
        # Shape : (B, H) => (B, 50)
        # x = self.dropout(x)
        
        # Shape : (B, 32)
        # x = self.dense1(x)
        # Shape : (B, O) // O = output => (B, 1)
        x = self.dense2(x)

        return x
    
    def save(self, fn):
        ''' Save the model to a file.
        
        Args:
            fn (str): filename.
        '''
        torch.save(self.state_dict(), fn)
    
    def load(self, fn):
        ''' Load the model from a file.
        
        Args:
            fn (str): filename.
        '''
        state_dict = torch.load(fn)
        self.load_state_dict(state_dict)