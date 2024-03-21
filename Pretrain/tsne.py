import sys
sys.path.append(sys.path[0].replace('report/visualize', ''))
import os
import numpy as np

from sklearn.manifold import TSNE
dirname, filename = os.path.split(os.path.abspath(__file__))


def get_tsne_results():
    feature_dict = {}
    cxr14 = np.load('img_cxr14.npy')
    cxp = np.load('img_cxp.npy')
    vindr = np.load('img_vindr.npy')
    feature_matrix = np.concatenate([cxr14, cxp, vindr])
    # 每个源的样本特征拼起来


    index0 = np.load('index0_1.npy')
    index1 = np.load('index1_1.npy')
    index2 = np.load('index2_1.npy')

    domain_label_list = np.concatenate([index0, index1, index2])
    # 每个源 domain label拼起来， 0,1,2,3分别代表cxr14, cxp, vindr
    
    tsne_data, tsne_model = tsne_evaluation(feature_matrix)
    feature_dict['tsne_data'] = tsne_data

    feature_dict['total_domain_label'] = domain_label_list
    
    np.save('tsne_results_1028.npy', feature_dict, allow_pickle=True)
    
    return feature_dict

def tsne_evaluation(feature_matrix):
    tsne_model = TSNE(n_components=2, init='pca', random_state=0)
    # 2维
    tsne_model.fit_transform(feature_matrix)
    return tsne_model.embedding_, tsne_model
        
if __name__ == "__main__":
    import time
    
    # for log_dir in log_list:
    feature_dict = get_tsne_results()
    bt_feature = feature_dict['tsne_data']

