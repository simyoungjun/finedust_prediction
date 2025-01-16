#%
import os
import pandas as pd

#####대화형으로 실행


#%
###Load data

folder_path = '../fineDust_data'
files_in_folder = os.listdir(folder_path)

dataframes = []
file_paths = []

for file_name in files_in_folder:
    if file_name.endswith('.csv'):
        file_path = os.path.join(folder_path, file_name)
        file_paths.append(file_path)
        #엑셀을 전부 읽어온 상태로 저장할 것인지
        # df = pd.read_csv(file_path)
        # dataframes.append(df)


# for idx, df in enumerate(dataframes):
#     print(f"DataFrame {idx + 1}:")
#     print(df.head())  # Display the first few rows of each DataFrame
#     print('-' * 40)

#각 년도별로 df 생성 (현재는 덮어씀)
for idx, file_path in enumerate(file_paths):
    print(f"file_name {idx + 1}:")
    df = pd.read_csv(file_path, encoding='cp949')
    df = df.fillna(method='ffill')
    print(df.head())  # Display the first few rows of each DataFrame
    print('-' * 40)
#%
###2022 data, 지역구 별 좌표 생성
file_2022csv = file_paths[4]

#df 는 2022
districts = df['구분'].unique()
districts = districts[districts != '평균']
print(districts)

#지역구별 좌표 값 (이름 사전순)
coordinates=[[337, 216],[437, 195],[327,88],[137, 143],[254, 292],[379, 190],[147, 272], [193, 297],[371, 35],[337, 25],[370, 153],[283,246],[256,178],[259,159],[311,234],[343,159],[305,98],[429,243],[189,191],[216,226],[285,180],[230,107],[280,103],[320,165],[392,134]]

print(len(coordinates))
print(len(districts))
def district_coordinate(coordinate):
    x = coordinate[0]
    y = coordinate[1]
    text_positions = [
        (x, y),  # (x, y) coordinates of the first text
        (x, y+8),  # (x, y) coordinates of the second text
        # Add more coordinates as needed
    ]
    return text_positions

###지역구 별 좌표 dict
districts_position = {}
for c in zip(districts, coordinates):
    print(c)
    districts_position[c[0]]= district_coordinate(c[1])


print(districts_position)


# districts_position['용산구'] = district_coordinate(280,190)
# print(districts_position['용산구'])

#%
### 지역구별 미세먼지 농도 plot (잘 나오는지 확인하기 위한 지역별 매칭 plot)

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
map_image_path = './서울시지도.jpg'
map_image = Image.open(map_image_path)

def annotate_map_with_text(image, positions, text_value, text_color='red', font=ImageFont.truetype("./gulim.ttc", size=10)):
    draw = ImageDraw.Draw(image)

    for position, value in zip(positions, text_value):
        for p in position:
            x, y = p
            value = str(value)
            print(value)
            print(x)
            print(y)
            draw.text((x, y), value, fill=text_color, font=font)
    return image

# text_value = [['Value 1', 'Value 2']]
text_value = districts_position.keys()
text_positions = districts_position.values()


# Annotate the map with text
annotated_map = annotate_map_with_text(map_image, text_positions, text_value)

# Display the annotated map
plt.figure(figsize=(16, 12))
plt.imshow(annotated_map)
plt.axis('off')  # Hide axis labels
plt.show()


#%
### 지역/일자 별로 PM 값들 정리.
import numpy as np

print(df.head())
print('districts : ', districts)
date_li = df['일시'].unique()
print(len(date_li))

data_li = []
for district in districts:
    df_district = df[df['구분'] == district]
    PMs_district = [df_district['미세먼지(PM10)'], df_district['초미세먼지(PM2.5)']]
    data_li.append(PMs_district)

data_arr = np.array(data_li).transpose((2,0,1))
print(data_arr.shape)
#%
### 지역구별 미세먼지 농도 plot

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
map_image_path = './서울시지도.jpg'


def annotate_map_with_PMs(PM_values, text_colors= ['red','blue']):
    map_image = Image.open(map_image_path)
    draw = ImageDraw.Draw(map_image)
    positions = districts_position.values()
    
    for position, PMs_dist in zip(positions, PM_values):
        for p, PM_value, text_color in zip(position, PMs_dist, text_colors):
            x, y = p
            # print(PM_value)
            # print(x)
            # print(y)
            draw.text((x, y), str(int(PM_value)), fill=text_color, font_size=10)
    return map_image

# Display the annotated map
def show_image():
    plt.figure(figsize=(16, 12))
    plt.imshow(annotated_map)
    plt.savefig('./data_analysis_fig/data_analysis_'+str(i)+'.png')
    plt.axis('off')  # Hide axis labels
    plt.show()

#%
PM_values = data_arr[5]
print(PM_values.shape)
# PM_values = districts_position.keys()

# Annotate the map with PMs
annotated_map = annotate_map_with_PMs(PM_values)
show_image()
# print(districts)
#%

for i in range(100):
    PM_values = data_arr[i]
    annotated_map = annotate_map_with_PMs(PM_values)
    show_image()
    
#%
