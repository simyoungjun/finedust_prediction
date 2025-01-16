#%
#분석
from train import *
import matplotlib.pyplot as plt



if __name__ == '__main__':
    
    # device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    # print(torch.cuda.is_available())
    # print(torch.cuda.get_device_name())
    
    file_path_for_all_districts = 'C:/Users/sim/Lab/심영준_포항공대/미세먼지/fineDust_data/2022.csv'
    df_for_columns = pd.read_csv(file_path_for_all_districts, encoding='cp949')
    
    try:
        train_path = './Datasets/preprocessed_2020-2021.csv'
        test_path = './Datasets/preprocessed_2022.csv'
        train_df = pd.read_csv(train_path, encoding='utf8')
        test_df = pd.read_csv(test_path, encoding='utf8')
        
        
    except:
        train_path = './custom/Datasets/preprocessed_2020-2021.csv'
        test_path = './custom/Datasets/preprocessed_2022.csv'
        train_df = pd.read_csv(train_path, encoding='utf8')
        test_df = pd.read_csv(test_path, encoding='utf8')
        
    all_districts = df_for_columns['구분'].unique()
    all_districts = all_districts[all_districts != '평균']

    # df = pd.read_csv(train_path, encoding='utf8')
    districts_list = all_districts
    # districts_list = ['중구', '종로구', '성동구', '동대문구', '성북구', '용산구', '마포구', '서대문구']
# #%
#     print(train_df.columns[1::])
#     districts_list = ['중구', '종로구', '성동구', '동대문구', '성북구', '용산구', '마포구', '서대문구']

#% 
train_selected_df = select_feature(train_df, districts_list)
test_selected_df = select_feature(test_df, districts_list)

train_ratio = 0.85
train_size = int(len(train_selected_df)*train_ratio//24)*24
# val_size = int(len(selected_df)*0.2)
train = np.array(train_selected_df[:train_size])
val = np.array(train_selected_df[train_size::])
test = np.array(test_selected_df)

#scaler
# scaler = MinMaxScaler()
# scaler.fit(train)

scaler = CumtomMinMaxScaler()
scaler.fit(train[:,1:])


#%
print(test.shape)

#%
# # 지역구 별 미세먼지 전체 plot

# # 한글 폰트 사용을 위해서 세팅
# from matplotlib import font_manager, rc
# font_path = "C:/Windows/Fonts/NGULIM.TTF"
# font = font_manager.FontProperties(fname=font_path).get_name()
# rc('font', family=font)


# for i in range(1, test.shape[1]):
#     # print(districts_list[i-1])
#     plt.figure()
#     plt.plot(test[:,i], color='blue', label='Ground Truth')
#     plt.legend()
#     plt.title(test_selected_df.columns[i])
#     plt.xlabel('Date')
#     plt.ylabel('Value')
#     # plt.ylim(0, 100)
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig('../data_analysis_fig/all_distributions/test_'+str(test_selected_df.columns[i])+'.png')
#     plt.show()

#%
# correlation 계산
corr_dict = {}
for i in range(1,train.shape[1]):
    
    # series1 = pd.Series(test[:,1])
    # series2 = pd.Series(test[:,i])
    # print(series1)
    # print(series2)
    correlation = np.corrcoef(train[:,1].astype(np.float64), train[:,i].astype(np.float64))
    # print(train_df.columns[0])
    print(str(i),'. ',train_df.columns[47], ' - ', train_df.columns[i], 'corr: ', np.round(correlation[0,1],3))
    corr_dict[train_df.columns[i]] = np.round(correlation[0,1],3)

corr_dict_sorted = {k: v for k, v in sorted(corr_dict.items(), key=lambda item: item[1], reverse = True)}

# corr_dict_sorted = sorted(corr_dict.values())
max_key_length = max(len(key) for key in corr_dict_sorted)

# print('중구 10과 나머지 전체 Corr\n')
for key, value in corr_dict_sorted.items():
    print(f'{key.ljust(max_key_length)}: {value}')

#%
#% 
# # fft 분석 Example time-domain signal
# sample_rate = 1000  # Sampling rate in Hz
# duration = 1  # Duration of the signal in seconds
# t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)  # Time values
# signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)  # Example signal with two frequencies

# # for i in range(train.shape[1]):
# for i in range(2):
    
#     fft_result = np.fft.fft(train[:,i])
#     # fft_freq = np.fft.fftfreq(len(fft_result), 3600)[len(fft_result)//2::]
#     fft_freq = np.fft.fftfreq(len(fft_result), 1/1000)
#     print(len(fft_result))
#     # amplitude_spectrum = np.abs(fft_result)
#     amplitude_spectrum = np.abs(fft_result)[len(fft_result)//4:len(fft_result)//2]/len(fft_result)
    
#     amplitude_spectrum[0] = 0
#     log_amplitude_spectrum = 10 * np.log10(amplitude_spectrum)
    
#     plt.figure(figsize=(10, 6))
#     # plt.plot(fft_freq, amplitude_spectrum)
#     plt.plot(amplitude_spectrum)
    
#     plt.xlabel('Frequency (Hz)')
#     plt.ylabel('Amplitude')
#     plt.title('Amplitude Spectrum')
#     plt.grid()
#     plt.show()