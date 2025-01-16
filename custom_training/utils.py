from config import Arguments as args
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_squared_error

def to_numpy_cpu(x):
    return x.cpu().detach().numpy()

def plot_graph(date, pred, y, idx, axes=None, sub_idx=None, y_r=False ):
    n_cols = args.pred_graph_ncols
    # First subplot
    # plt.plot(scaler.inverse_transform(pred), marker='o', color='blue')
    # plt.plot(scaler.inverse_transform(y), marker='x', color='red')
    date = date
    rmse = np.sqrt(mean_squared_error(y, pred))
    # axes[sub_idx//n_cols, sub_idx%n_cols].plot(date, pred, marker='o', color='red', label='Pred')
    # axes[sub_idx//n_cols, sub_idx%n_cols].plot(date, y, marker='x', color='blue', label='Ground Truth')
    axes[sub_idx//n_cols, sub_idx%n_cols].plot(pred, marker='o', color='red', label='Pred')
    axes[sub_idx//n_cols, sub_idx%n_cols].plot(y, marker='x', color='green', label='Ground Truth(MA)')
    axes[sub_idx//n_cols, sub_idx%n_cols].plot(y_r, linestyle='dotted', color='blue', label='Ground Truth')

    axes[sub_idx//n_cols, sub_idx%n_cols].legend()
    axes[sub_idx//n_cols, sub_idx%n_cols].set_title('PM10 rmse: '+str(rmse))
    # axes[sub_idx//n_cols, sub_idx%n_cols].set_xlabel('Date')
    # axes[sub_idx//n_cols, sub_idx%n_cols].set_xticklabels(date, rotation=320)
    axes[sub_idx//n_cols, sub_idx%n_cols].set_ylabel('Value')
    axes[sub_idx//n_cols, sub_idx%n_cols].set_ylim(min(y)-10, max(y)+10)
    axes[sub_idx//n_cols, sub_idx%n_cols].grid(True)
    # axes[sub_idx].show(block=args.block)
    # axes[sub_idx].savefig(args.save_fig_path+'graph_plot'+str(idx)+'.png')
    
    return plt

def plot_histogram(pred, y, title = '', axes=None, sub_idx = None):
    plt.hist(y, bins=20, color='skyblue', edgecolor='black')
    plt.xlabel('Value')  # x 축 레이블
    plt.ylabel('Frequency')  # y 축 레이블
    plt.title('Histogram Example')  # 그래프 제목
    plt.grid(True) 
    
def scatter_plot_result(pred, y, title, xy_lim = [0, args.max_num+10], axes=None, sub_idx = None):

    # plt.figure()
    marker_size = 20  # Adjust this value as needed
    border_color = 'black'
    # Create a scatter plot
    # _ = np.linspace(pred.min(), pred.max(), 100)
    _ = np.linspace(0, y.max()+10, 200)
    
    rmse = np.round(np.sqrt(mean_squared_error(y, pred)),3)
    mae = np.round(np.mean(np.abs(y - pred)), 3)
    cor = np.round(np.corrcoef(y.flatten(), pred.flatten())[0, 1], 3)
    
    axes[sub_idx].scatter(y, pred, s=marker_size, edgecolors=border_color, linewidth=1)
    axes[sub_idx].plot(_, _, label='Graph Line', color='red', linestyle='--', linewidth=3)

    # Add labels and a title
    axes[sub_idx].set_xlim(xy_lim[0],xy_lim[1])
    axes[sub_idx].set_ylim(xy_lim[0],xy_lim[1])
    axes[sub_idx].set_xlabel('Observed PM10')
    axes[sub_idx].set_ylabel('Predicted PM10')
    axes[sub_idx].set_title('RMSE: '+str(rmse)+' MAE:'+str(mae)+' Cor: '+str(cor))

    # Show the plot
    # axes[sub_idx].show(block=args.block)
    # axes[sub_idx].savefig(args.save_fig_path+title+'.png')
    
    return plt

def max_time_heatmap(pred, y, title, xy_lim = [0, 310], axes=None, sub_idx = None):
    
    # 2x3 크기의 빈 2차원 행렬 생성
    
    max_time_matrix = np.zeros((24, 24))

    for i in range(len(pred)):
        x_idx = pred[i]
        y_idx = y[i]
        max_time_matrix[x_idx, y_idx] = max_time_matrix[x_idx, y_idx] + 1

    
    axes[sub_idx] = sns.heatmap(max_time_matrix, annot=True, cmap='coolwarm')
    axes[sub_idx].invert_yaxis()
    axes[sub_idx].set_title('Heatmap Example')  # 그래프 제목
    axes[sub_idx].set_xlabel('Observed PM10')
    axes[sub_idx].set_ylabel('Predicted PM10')  # y축 레이블

    # axes[sub_idx].show(block=args.block)
    # axes[sub_idx].savefig(args.save_fig_path+title+'.png')

    return plt
    
def train_log_fig(xs, ys, axes=None, sub_idx = None):
    # plt.figure(figsize=(8, 6))
    plt.plot(xs, ys[0], label='Train Loss')
    plt.plot(xs, ys[1], label='Validation Loss')
    # 그래프 제목 및 레이블 설정
    plt.title('Train Loss and Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show(block=args.block) 
    plt.savefig(args.save_fig_path)
    
def merge_results(data):
    
    merging_num = 24//args.ow
    if args.ow == 24:
        merged_data = data
    else:
        merged_data = [np.concatenate(data[i:i+merging_num]) for i in range((24-args.iw)//args.ow, len(data), merging_num)]
    
    return np.array(merged_data)


from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_scatter(TRUTH_DATA,FORECAST_DATA, vlim, fig = None, axes=None, sub_idx = None):

    USE = ((TRUTH_DATA>0) | (FORECAST_DATA>0))
    TRUTH_DATA = TRUTH_DATA[USE]
    FORECAST_DATA = FORECAST_DATA[USE]

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 20

    ###Calculate Correlation
    TRUTH_DATA_CORR = TRUTH_DATA.copy()
    FORECAST_DATA_CORR = FORECAST_DATA.copy()
    effective = np.where((TRUTH_DATA_CORR>0) & (FORECAST_DATA_CORR>0))
    TRUTH_DATA_CORR = TRUTH_DATA_CORR[effective].reshape([-1])
    FORECAST_DATA_CORR = FORECAST_DATA_CORR[effective].reshape([-1])
    corr = np.round(np.corrcoef(TRUTH_DATA_CORR,FORECAST_DATA_CORR)[0][1],2)
    text = "CORR=" + str(corr)

    ###Heatmap Plot
    heatmap, xedges, yedges = np.histogram2d(FORECAST_DATA_CORR,TRUTH_DATA_CORR,range=[[vlim[0],vlim[1]],[vlim[0],vlim[1]]],bins=60,density=False)
    extent = [xedges[0],xedges[-1],yedges[0],yedges[-1]]
    h = plt.imshow(np.log10(heatmap),extent=extent,origin="lower",cmap="jet")
    plt.plot(np.arange(vlim[0],vlim[1]+100),np.arange(vlim[0],vlim[1]+100),c='black')
    axes[sub_idx].set_aspect('equal', 'datalim')
    #AXIS LABELs
    axes[sub_idx].set_xlabel("TRUTH[dBZ]")
    axes[sub_idx].set_ylabel("FORECAST[dBZ]")
    plt.text(10,190,text)
    axes[sub_idx].grid(which='major',color='black',linestyle='-',linewidth=0.1)
    tick = np.arange(vlim[0],vlim[1]+0.1,20)
    plt.xticks(tick,tick.astype(str))
    plt.yticks(tick,tick.astype(str))
    axes[sub_idx].set_xlim(vlim[0],vlim[1])
    axes[sub_idx].set_ylim(vlim[0],vlim[1])
    
    # #COLORBAR SETTTING
    divider = make_axes_locatable(axes[sub_idx])
    ax_cb = divider.new_horizontal(size="2%", pad=0.05)
    fig.add_axes(ax_cb)
    cbar = plt.colorbar(h, cax=ax_cb)
    cbar.set_label('log$_{10}$(Number of Data)')
    #SAVE a IMAGE
    # plt.savefig(savename + "_heat.png")
    plt.show()
    
    # plt.clf()
    # plt.close()

import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
def kde_plot(y, pred, vlim, fig = None, axes=None, sub_idx = None):
    
    cmap = sns.color_palette("Reds", as_cmap=True)
    new_cmap = LinearSegmentedColormap.from_list(
        'IntenseReds',
        cmap(np.linspace(0, 1, 256))**2, # Squaring to reduce the lightness, making the colors more intense
        N=256
)
    # KDE plot on the second subplot
    # sns.kdeplot(x=y.flatten(), y=pred.flatten(), cmap=new_cmap, fill=True, ax=axes[sub_idx], bw_adjust=.4)
    # axes[sub_idx].set_title('KDE Plot of PM10')
    # axes[sub_idx].set_xlabel('Observed PM10')
    # axes[sub_idx].set_ylabel('Predicted PM10')
    
    # # Add a color bar for the KDE plot
    # cbar = plt.colorbar(axes[sub_idx].collections[0], ax=axes[sub_idx])
    # cbar.set_label('Density')
    
    hb = axes[sub_idx].hexbin(x=y.flatten(), y=pred.flatten(), gridsize=50, cmap='Reds', norm=mcolors.LogNorm())
    # counts, xedges, yedges, im = axes[sub_idx].hist2d(x=y.flatten(), y=pred.flatten(), bins=40)
    # Add a color bar for the hexbin plot
    cbar = plt.colorbar(hb)
    cbar.set_label('Log(Count)')

    # Add labels and title
    axes[sub_idx].set_title('Hexbin Plot of PM10 Predictions')
    axes[sub_idx].set_xlabel('Observed PM10')
    axes[sub_idx].set_ylabel('Predicted PM10')

    # Show the plot
    # plt.show()