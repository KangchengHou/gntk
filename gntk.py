import math
import numpy as np
import scipy as sp

class GNTK(object):
    """
    implement the Graph Neural Tangent Kernel
    """
    def __init__(self, num_layers, num_mlp_layers, jk, scale):
        """
        num_layers: number of layers in the neural networks (including the input layer)
        num_mlp_layers: number of MLP layers
        jk: a bool variable indicating whether to add jumping knowledge
        scale: the scale used aggregate neighbors [uniform, degree]
        """
        self.num_layers = num_layers
        self.num_mlp_layers = num_mlp_layers
        self.jk = jk
        self.scale = scale
        assert(scale in ['uniform', 'degree'])
    
    def __next_diag(self, S):
        """
        go through one normal layer, for diagonal element
        S: covariance of last layer
        """
        diag = np.sqrt(np.diag(S))
        S = S / diag[:, None] / diag[None, :]
        S = np.clip(S, -1, 1)
        # dot sigma
        DS = (math.pi - np.arccos(S)) / math.pi
        S = (S * (math.pi - np.arccos(S)) + np.sqrt(1 - S * S)) / np.pi
        S = S * diag[:, None] * diag[None, :]
        return S, DS, diag

    def __adj_diag(self, S, adj_block, N, scale_mat):
        """
        go through one adj layer
        S: the covariance
        adj_block: the adjacency relation
        N: number of vertices
        scale_mat: scaling matrix
        """
        return adj_block.dot(S.reshape(-1)).reshape(N, N) * scale_mat

    def __next(self, S, diag1, diag2):
        """
        go through one normal layer, for all elements
        """
        S = S / diag1[:, None] / diag2[None, :]
        S = np.clip(S, -1, 1)
        DS = (math.pi - np.arccos(S)) / math.pi
        S = (S * (math.pi - np.arccos(S)) + np.sqrt(1 - S * S)) / np.pi
        S = S * diag1[:, None] * diag2[None, :]
        return S, DS
    
    def __adj(self, S, adj_block, N1, N2, scale_mat):
        """
        go through one adj layer, for all elements
        """
        return adj_block.dot(S.reshape(-1)).reshape(N1, N2) * scale_mat
      
    def diag(self, g, A):
        """
        compute the diagonal element of GNTK for graph `g` with adjacency matrix `A`
        g: graph g
        A: adjacency matrix
        """
        N = A.shape[0]
        if self.scale == 'uniform':
            scale_mat = 1.
        else:
            scale_mat = 1. / np.array(np.sum(A, axis=1) * np.sum(A, axis=0))

        diag_list = []
        adj_block = sp.sparse.kron(A, A)

        # input covariance
        sigma = np.matmul(g.node_features, g.node_features.T)
        sigma = self.__adj_diag(sigma, adj_block, N, scale_mat)
        ntk = np.copy(sigma)
		
        
        for layer in range(1, self.num_layers):
            for mlp_layer in range(self.num_mlp_layers):
                sigma, dot_sigma, diag = self.__next_diag(sigma)
                diag_list.append(diag)
                ntk = ntk * dot_sigma + sigma
            # if not last layer
            if layer != self.num_layers - 1:
                sigma = self.__adj_diag(sigma, adj_block, N, scale_mat)
                ntk = self.__adj_diag(ntk, adj_block, N, scale_mat)
        return diag_list

    def gntk(self, g1, g2, diag_list1, diag_list2, A1, A2):
        """
        compute the GNTK value \Theta(g1, g2)
        g1: graph1
        g2: graph2
        diag_list1, diag_list2: g1, g2's the diagonal elements of covariance matrix in all layers
        A1, A2: g1, g2's adjacency matrix
        """
        
        n1 = A1.shape[0]
        n2 = A2.shape[0]
        
        if self.scale == 'uniform':
            scale_mat = 1.
        else:
            scale_mat = 1. / np.array(np.sum(A1, axis=1) * np.sum(A2, axis=0))
        
        adj_block = sp.sparse.kron(A1, A2)
        
        jump_ntk = 0
        sigma = np.matmul(g1.node_features, g2.node_features.T)
        jump_ntk += sigma
        sigma = self.__adj(sigma, adj_block, n1, n2, scale_mat)
        ntk = np.copy(sigma)
        
        for layer in range(1, self.num_layers):
            for mlp_layer in range(self.num_mlp_layers):
                sigma, dot_sigma = self.__next(sigma, 
                                    diag_list1[(layer - 1) * self.num_mlp_layers + mlp_layer],
                                    diag_list2[(layer - 1) * self.num_mlp_layers + mlp_layer])
                ntk = ntk * dot_sigma + sigma
            jump_ntk += ntk
            # if not last layer
            if layer != self.num_layers - 1:
                sigma = self.__adj(sigma, adj_block, n1, n2, scale_mat)
                ntk = self.__adj(ntk, adj_block, n1, n2, scale_mat)
        if self.jk:
            return np.sum(jump_ntk) * 2
        else:
            return np.sum(ntk) * 2
