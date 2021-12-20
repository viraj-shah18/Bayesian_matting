import numpy as np
# Implementation of orchard bouman clustering - Taken from the github repo linked in references
# there was no direct API available for Orchard-Boumann clustering available in python libraries


class Node(object):
    def __init__(self, matrix, w):
        W = np.sum(w)
        if W == 0:
            # print(w)
            print(":hey")
        self.w = w
        self.X = matrix
        self.left = None
        self.right = None
        self.mu = np.einsum("ij,i->j", self.X, w) / W
        diff = self.X - np.tile(self.mu, [np.shape(self.X)[0], 1])
        t = np.einsum("ij,i->ij", diff, np.sqrt(w))
        self.cov = (t.T @ t) / W + 1e-5 * np.eye(3)
        self.N = self.X.shape[0]
        V, D = np.linalg.eig(self.cov)
        self.lmbda = np.max(np.abs(V))
        self.e = D[np.argmax(np.abs(V))]


# S is required vector whose mean and coviarance matrix is given as output for each cluster
# w is weights vector (gaussian weighted - to give higher importance to nearby pixels)
def clustFunc(S, w, minVar=0.05):
    mu, sigma = [], []
    nodes = []
    nodes.append(Node(S, w))

    while max(nodes, key=lambda x: x.lmbda).lmbda > minVar:
        nodes = split(nodes)

    for i, node in enumerate(nodes):
        mu.append(node.mu)
        sigma.append(node.cov)

    return np.array(mu), np.array(sigma)


def split(nodes):
    idx_max = max(enumerate(nodes), key=lambda x: x[1].lmbda)[0]
    C_i = nodes[idx_max]
    idx = C_i.X @ C_i.e <= np.dot(C_i.mu, C_i.e)
    C_a = Node(C_i.X[idx], C_i.w[idx])
    C_b = Node(C_i.X[np.logical_not(idx)], C_i.w[np.logical_not(idx)])
    nodes.pop(idx_max)
    nodes.append(C_a)
    nodes.append(C_b)
    return nodes
