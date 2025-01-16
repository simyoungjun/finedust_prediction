import numpy as np
from config import Arguments as args

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from losses import min_max_weight, max_weight

class CustomDataset(Dataset):
    def __init__(self, y, scaler, input_window=24, output_window=24, stride= 24, train_scale = True):
        date_time = y[:,0].astype(str)
        y = y[:,1:]
        
        if scaler == None:
            pass
            
        elif scaler.segment == False:
            y = scaler.transform(y)
        
            
        L = y.shape[0]
        num_features = y.shape[1]
        num_samples = (L - input_window - output_window) // stride + 1

        # X = np.zeros([input_window, num_samples])
        # Y = np.zeros([output_window, num_samples])
        X = np.zeros([num_samples, input_window, num_features])
        Y = np.zeros([num_samples, output_window])
        Date_time_y = []
        for i in tqdm(np.arange(num_samples)):
            start_x = stride*i
            end_x = start_x + input_window
            start_y = stride*i + input_window
            end_y = start_y + output_window
            X[i] = y[start_x:end_x]
            Y[i] = y[start_y:end_y,0]
            Date_time_y.append(date_time[start_y:end_y])
            
        diff_weight = max_weight(X)
        # diff_weight = min_max_weight(X)

        if scaler.segment == True:
            for i in tqdm(np.arange(num_samples)):
                scaler.X_min = X[i].min(axis=0)
                scaler.X_max = X[i].max(axis=0)
                scaler.X_min_segments.append(scaler.X_min)
                scaler.X_max_segments.append(scaler.X_max)
                
                X[i] = scaler.transform(X[i])
                Y[i] = scaler.transform_for_GT(Y[i])
                
        
        if train_scale != True: #val 이나 test set 에서만 inverse할 것이기 때문에 이게 저장되어야함. batch개수 1이기 때문에 batch 차원은 고려 안해도됨-> 2차원
            scaler.X_min_segments = np.array(scaler.X_min_segments)
            scaler.X_max_segments = np.array(scaler.X_max_segments)
        else:
            scaler.X_min_segments = []
            scaler.X_max_segments = []
            
        self.date_time_y = np.array(Date_time_y)
        self.x = X
        self.y = Y
        self.diff_weight = diff_weight
        
        self.len = len(X)
    def __getitem__(self, i):
        #이부분에서 date_time의 type이 str/O 면 배치로더가 안됨. 이거 해결해라
        return i, self.x[i], self.y[i]
    def __len__(self):
        return self.len
    
class CumtomMinMaxScaler:
    # def __init__(self, X):

        
    def fit(self, X, max_num = None, segment = False):
        self.X_min = X.min(axis=0)
        self.X_max = X.max(axis=0)
        self.X_max_segments = []
        self.X_min_segments = []
        self.segment = segment
        self.max_num = max_num
        
        if max_num != None:
            if self.segment == False:
                self.X_max[self.X_max>max_num] = max_num
            else:
                pass
        
        

    def constraint_max(self, data):
        if self.max_num != None:
            data[data > self.max_num] = self.max_num
        return data
    
    def transform(self, data): # transform 할때 segment 할때는 각 세그먼트가 data에 들어오기 때문에 세그먼트 안했을때랑 코드 같이 쓸 수 있음.
        if self.max_num!=None and self.segment==False:
            data = self.constraint_max(data)
        
        data_scaled = (data - self.X_min ) / (self.X_max - self.X_min)
        return data_scaled
    
    def transform_for_GT(self, data, i=0):
        data_scaled = (data - self.X_min[i] ) / (self.X_max[i] - self.X_min[i])
        return data_scaled
    
    def inverse_transform(self, data_scaled, i=-1):
        data = data_scaled * (self.X_max - self.X_min) + self.X_min
        return data
    
    def inverse_transform_for_GT(self, data_scaled, i=-1): #val, test의 경우 batch개수 1개임.
        data = data_scaled
        if self.segment == True: #self.X_max가 2차원
            for sample_i in range(data_scaled.shape[0]): 
                data[sample_i] = data_scaled[sample_i] * (self.X_max_segments[sample_i, i] - self.X_min_segments[sample_i, i]) + self.X_min_segments[sample_i, i]
        else: # self.X_max가 1차원.
            data = data_scaled * (self.X_max[i] - self.X_min[i]) + self.X_min[i]
        return data
    
    
def create_sequences(data, input_seq_length, output_seq_length):
    xs = []
    ys = []
    for i in tqdm(range(len(data)- max(input_seq_length, output_seq_length))):
        x = data.iloc[i:(i+input_seq_length)]
        y = data.iloc[(i+input_seq_length):(i+input_seq_length+output_seq_length)]
        xs.append(x)
        ys.append(y)
    
    xs_arr = list(map(df_list_to_array, xs))
    ys_arr = list(map(df_list_to_array, ys))
        
    return np.array(xs_arr[:10]), np.array(ys_arr[:10])


def df_list_to_array(df):
    return np.array(df)

def column_PM25(district):
    return district+'_2.5PM'

def column_PM10(district):
    return district+'_10PM'
    
    

def select_feature(data, districts_list, PMs = [True, True]):
    column_selected = ['date']

    if PMs[0] == True:
        column_selected = column_selected + list(map(column_PM10, districts_list))
    if PMs[1] == True:
        column_selected = column_selected + list(map(column_PM25, districts_list))

    return data[column_selected]