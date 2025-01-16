#%
import os
import pandas as pd
import numpy as np



def load_finedust(folder_path):
###Load data

    files_in_folder = os.listdir(folder_path)

    dataframes = []
    file_paths = []
    file_names = []
    for file_name in files_in_folder:
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            file_paths.append(file_path)
            file_names.append(file_name)
            #엑셀을 전부 읽어온 상태로 저장할 것인지
            # df = pd.read_csv(file_path)
            # dataframes.append(df)


    # for idx, df in enumerate(dataframes):
    #     print(f"DataFrame {idx + 1}:")
    #     print(df.head())  # Display the first few rows of each DataFrame
    #     print('-' * 40)

    #각 년도별로 df 생성 (현재는 덮어씀)
    df_list = []
    for idx, file_path in enumerate(file_paths):
        print(f"file_name {idx + 1}:")
        df = pd.read_csv(file_path, encoding='cp949')
        df_list.append(df)
        print(df.head())  # Display the first few rows of each DataFrame
        print('-' * 40)
    
    return df_list, file_names

def union_data(df_list):
    new_df_list = []
    for df in df_list:
        new_df = preprocess_finedust(df)
        new_df_list.append(new_df)
    union_data = pd.concat(new_df_list, axis=1)
    return union_data

#%
def preprocess_finedust(df):
    df = df[df['구분'] != '평균']
    df.head()
    df = df.set_index('일시')
    df.index = df.index.rename('date')

    districts = df['구분'].unique()
    districts = districts[districts != '평균']
    print(districts)

    # print(df[df['구분'] == '강남구'][:30])


    # print(df.head())
    # print('districts : ', districts)
    # date_li = df['일시'].unique()
    # print(len(date_li))

    new_df_dict = {}
    new_df = pd.DataFrame(new_df_dict)
    for district in districts:
        district_10PM = df[df['구분'] == district]['미세먼지(PM10)']
        try:
            district_025PM = df[df['구분'] == district]['초미세먼지(PM25)']
        except:
            district_025PM = df[df['구분'] == district]['초미세먼지(PM2.5)']
        new_df[district+'_10PM'] = district_10PM
        new_df[district+'_2.5PM'] = district_025PM
    
    new_df = new_df.iloc[::-1]

    first_row_mean = new_df.iloc[0].mean()
    last_row_mean = new_df.iloc[-1].mean()
    
    n = new_df.iloc[0,0]
    print(np.isnan(n))
    # 첫 번째 행의 NaN 값을 1로 채우기
    for column_idx in range(len(new_df.columns)):
        if np.isnan(new_df.iloc[0,column_idx]):
            new_df.iloc[0,column_idx] = first_row_mean
        if np.isnan(new_df.iloc[-1,column_idx]):
            new_df.iloc[-1,column_idx] = last_row_mean
    #Nan 값을 바로 이전의 값으로 채워줌
    # new_df = new_df.fillna(method='ffill')
    
    new_df.interpolate(method='linear', inplace=True)
    
        # df_district = df[df['구분'] == district]
        # PMs_district = [df_district['미세먼지(PM10)'], df_district['초미세먼지(PM2.5)']]
        # data_li.append(PMs_district)

    print(new_df[:20])

    #df 에 nan값 있는지확인
    has_nan = new_df.isna()  # 또는 has_nan = df.isnull()
    print(has_nan.sum().sum())
    # data_arr = np.array(data_li).transpose((2,0,1))
    # print(data_arr.shape)
    
    new_df = new_df.reset_index()
    print(new_df.columns)
    
    new_df['date'] = pd.to_datetime(new_df['date'])
    new_df['time_column'] = new_df['date'].dt.strftime('%H')
    new_df['date_column'] = new_df['date'].dt.date
    
    print(new_df[:30])
    
    # new_df = eliminate_missing_data(new_df[:40])
    
    # 24t
    counts = new_df['date_column'].value_counts()
    threshold = 24
    new_df_filtered = new_df[new_df['date_column'].map(counts) == threshold]
    
    counts = new_df_filtered['date_column'].value_counts()
    
    print(np.unique(counts))
    
    return new_df_filtered

# def eliminate_missing_data(df):
#     drop_index_list = []
#     drop_day = -1
#     for i in range(len(df)):
#         time_gt = (i - len(drop_index_list)) % 24
#         # s = int(df.loc[i,'time_column'])
#         if int(df.loc[i,'day_column']) == drop_day:
#             drop_index_list.append(i)
            
#         elif int(df.loc[i,'time_column']) != time_gt:
#             drop_day = df.loc[i,'day_column']
#             drop_index_list.append(i)
#             # s = int(df.loc[i,'time_column'])

#     df = df.drop(drop_index_list)
#     return df
#%


if __name__ == '__main__':
    folder_path = 'C:/Users/sim/Lab/심영준_포항공대/미세먼지/fineDust_data'
    df_list, file_names = load_finedust(folder_path)
    for i, file_name in enumerate(file_names):
        new_df = preprocess_finedust(df_list[i])
    # print(new_df.max())
    # print(new_df.min())
    # union_df = union_data(df_list)
        new_df.to_csv('./custom/Datasets/preprocessed_LinearInter_'+file_name)
        # new_df.to_csv('./Datasets/preprocessed_LinearInter_'+file_name)
        
#%
