from model import GAT_lays
import numpy as np
import scipy.sparse as sp
import pprint


def encode_onehot(all_labels):
    # 对所有的label 进行编号，再将编号转换成 one_hot向量
    print(all_labels.tolist())
    classes = sorted(list(set(all_labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    pprint.pprint(classes_dict)
    labels_onehot = np.array(list(map(classes_dict.get, all_labels)), dtype=np.int32)

    return labels_onehot


path = "./data/cora/"
dataset = "cora"
content = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
features = sp.csr_matrix(content[:, 1:-1], dtype=np.float32)
labels = encode_onehot(content[:, -1])

#
# print("cora content shape = ", content.shape)
#
# print("features  = ", features.shape)
#
# print("labels shape = ", labels.shape)

# print(content)

idx = np.array(content[:, 0], dtype=np.int32)
idx_map = {j:i for i, j in enumerate(idx)}
raw_edges = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
print("raw_edges = ", raw_edges)
edges=np.array(list(map(idx_map.get,raw_edges.flatten())),dtype=np.int32).reshape(raw_edges.shape)
#邻接矩阵
adj = sp.coo_matrix((np.ones(edges.shape[0]),(edges[:,0],edges[:,1])),shape=(labels.shape[0],labels.shape[0]))
