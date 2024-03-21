import sys
sys.path.append(sys.path[0].replace('report/visualize', ''))
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# color_list = ['#CBEDF9', '#FFECBD', '#C9B5DD'] 
# # 浅色 洪峰主图
# color_list = ['#A2DFF4', '#8DD34D', '#FFDE8B'] 

# 深色 主图

color_list = ['#FF0000', '#0000FF', '#008000', '#FFFF00', '#800080', '#FFA500', '#A52A2A', '#FFC0CB']

color_list = list(color_list)


def draw_tsne_picture(tsne_data, label):
    '''
    将相同label的数据画成一个颜色在一起
    '''
    label_list = np.unique(label)
    dataset_list = ['ChestX-ray14', 'CheXpert', 'VinDr-CXR']
    for label_id in label_list:
        selected_data = tsne_data[label==label_id, :]
        if label_id == 0:
            # print(selected_data.shape)
            # print(selected_data)
            # indices_1 = np.random.choice(selected_data.shape[0], int(selected_data.shape[0]/60), replace=False)

            # indices_2 = np.where((selected_data[:, 0] > -10) & (selected_data[:, 1] > -20))[0]
            # indices_3 = np.random.choice(indices_2, int(len(indices_2)/10), replace=False)

            # indices_4 = np.where((selected_data[:, 0] + selected_data[:, 1] > -20))[0]
            # indices_5 = np.random.choice(indices_4, int(len(indices_4)/40), replace=False)


            # indices = np.union1d(indices_1, indices_3)   
            # indices = np.union1d(indices, indices_5)
            # selected_data = tsne_data[label==label_id, :][indices, :]

            indices = np.random.choice(selected_data.shape[0], int(selected_data.shape[0]), replace=False)
            selected_data = tsne_data[label==label_id, :][indices, :]
        elif label_id == 1 or label_id == 2:
            indices = np.random.choice(selected_data.shape[0], int(selected_data.shape[0]), replace=False)
            selected_data = tsne_data[label==label_id, :][indices, :]

        plt.scatter(selected_data[:, 0], selected_data[:, 1], label=dataset_list[label_id], s=50, c=color_list[label_id])

    plt.axis('off')


def draw_paper_tsne_picture():
    save_dir = './'
    plt.figure(figsize=(18, 9))
    plt.subplots_adjust(wspace=0.0, hspace=0.0)

    feature_dict = np.load('tsne_results_1028.npy', allow_pickle=True).item()
    
    source_tsne = feature_dict['tsne_data']
    
    source_domain_label = feature_dict['total_domain_label']

    draw_tsne_picture(source_tsne, source_domain_label)
    # plt.title('Motivation', fontsize=32, 
    #         #   fontweight='heavy', 
    #           x=0.5, y=0.97)
    
    from matplotlib.lines import Line2D
    legend_elements = []
    dataset_list = ['ChestX-ray14', 'CheXpert', 'VinDr-CXR']
    for domain_name in np.unique(source_domain_label):
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label = dataset_list[domain_name], markerfacecolor=color_list[int(domain_name)], markersize=20))
    plt.legend(handles=legend_elements, loc=1, fontsize=24, handletextpad=0.1, bbox_to_anchor=(0.22, 0.32))
    plt.savefig(os.path.join(save_dir, f'motiva.png'), dpi=300, bbox_inches='tight')
    # plt.show()
    
    
if __name__ == "__main__":
    import matplotlib
    matplotlib.rcParams['text.usetex'] = False
    params={'font.family':'serif',
        'font.serif':'Times New Roman',
        'font.style':'normal',

        }
    matplotlib.rcParams.update(params)
    ''' paper 画图代码'''
    draw_paper_tsne_picture()
    