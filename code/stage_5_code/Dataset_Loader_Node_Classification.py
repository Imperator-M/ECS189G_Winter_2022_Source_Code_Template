"""
Concrete IO class for a specific dataset
"""

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

from code.base_class.dataset import dataset
import numpy as np
import scipy.sparse as sp
import torch


class Dataset_Loader(dataset):
    data = None
    dataset_name = None

    def __init__(self, seed=None, dName=None, dDescription=None):
        super(Dataset_Loader, self).__init__(dName, dDescription)

    def adj_normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -0.5).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx).dot(r_mat_inv)
        return mx

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert sparse matrix to sparse torch tensor"""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def encode_onehot(self, labels):
        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
        onehot_labels = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
        return onehot_labels

    def load(self):
        """Load citation network dataset"""
        print('Loading {} dataset...'.format(self.dataset_name))

        # load node data from file
        index_features_labels = np.genfromtxt("{}/node".format(self.dataset_source_folder_path), dtype=np.dtype(str))
        features = sp.csr_matrix(index_features_labels[:, 1:-1], dtype=np.float32)
        onehot_labels = self.encode_onehot(index_features_labels[:, -1])

        # load link data from file and build graph
        index = np.array(index_features_labels[:, 0], dtype=np.int32)
        index_map = {j: i for i, j in enumerate(index)}
        reverse_index_map = {i: j for i, j in enumerate(index)}
        edges_unordered = np.genfromtxt("{}/link".format(self.dataset_source_folder_path), dtype=np.int32)
        edges = np.array(list(map(index_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(onehot_labels.shape[0], onehot_labels.shape[0]), dtype=np.float32)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        norm_adj = self.adj_normalize(adj + sp.eye(adj.shape[0]))

        # convert to pytorch tensors
        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(np.where(onehot_labels)[1])
        adj = self.sparse_mx_to_torch_sparse_tensor(norm_adj)

        # You can put the following code in the setting class or leave it here.
        # These train, test, and val index are just examples.  Tune the sampling to your needs.

        # Code for randomly sampling dataset.
        num_train_per_class = 20
        num_test_per_class = 150
        num_classes = 7

        index_train = []
        index_test = []

        label_indices = {c: np.where(labels == c)[0] for c in range(num_classes)}

        for i in range(num_classes):
            # Randomly choose 20 indices for training.
            train_index = np.random.choice(label_indices[i], size=num_train_per_class, replace=False)
            # Randomly choose 150 indices for testing.
            test_index = np.random.choice(np.setdiff1d(label_indices[i], train_index), size=num_test_per_class, replace=False)
            # Add these indices to the training and testing index lists.
            index_train.extend(train_index)
            index_test.extend(test_index)


        if self.dataset_name == 'cora':
            #index_train = range(140)
            #index_test = range(200, 1200)
            index_val = range(1200, 1500)
        elif self.dataset_name == 'citeseer':
            #index_train = range(120)
            #index_test = range(200, 1200)
            index_val = range(1200, 1500)
        elif self.dataset_name == 'pubmed':
            #index_train = range(60)
            #index_test = range(6300, 7300)
            index_val = range(6000, 6300)
        # ---- cora-small is a toy dataset I handcrafted for debugging purposes ---
        elif self.dataset_name == 'cora-small':
            #index_train = range(5)
            index_val = range(5, 10)
            #index_test = range(5, 10)


        index_train = torch.LongTensor(index_train)
        index_val = torch.LongTensor(index_val)
        index_test = torch.LongTensor(index_test)
        # Get the training nodes and testing nodes.
        # train_x = features[index_train]
        # val_x = features[index_val]
        # test_x = features[index_test]
        # print(train_x, val_x, test_x)

        train_test_val = {'index_train': index_train, 'index_test': index_test, 'index_val': index_val}
        graph = {'node': index_map, 'edge': edges, 'X': features, 'y': labels, 'utility': {'A': adj, 'reverse_index': reverse_index_map}}
        return {'graph': graph, 'train_test_val': train_test_val}
