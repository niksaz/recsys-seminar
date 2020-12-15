from typing import List

import numpy as np
import lightgbm as lgb
import random
import math

from simanneal import Annealer


def countSplitNodes(tree):
    root = tree['tree_structure']
    def counter(root):
        if 'split_index' not in root:
            return 0
        return 1 + counter(root['left_child']) + counter(root['right_child'])
    ans = counter(root)
    return ans

def getItemByTree(tree, item='split_feature'):
    root = tree.raw['tree_structure']
    split_nodes = tree.split_nodes
    res = np.zeros(split_nodes+tree.raw['num_leaves'], dtype=np.int32)
    if 'value' in item or 'threshold' in item or 'split_gain' in item:
        res = res.astype(np.float64)
    def getFeature(root, res):
        if 'child' in item:
            if 'split_index' in root:
                node = root[item]
                if 'split_index' in node:
                    res[root['split_index']] = node['split_index']
                else:
                    res[root['split_index']] = node['leaf_index'] + split_nodes # need to check
            else:
                res[root['leaf_index'] + split_nodes] = -1
        elif 'value' in item:
            if 'split_index' in root:
                res[root['split_index']] = root['internal_'+item]
            else:
                res[root['leaf_index'] + split_nodes] = root['leaf_'+item]
        else:
            if 'split_index' in root:
                res[root['split_index']] = root[item]
            else:
                res[root['leaf_index'] + split_nodes] = -2
        if 'left_child' in root:
            getFeature(root['left_child'], res)
        if 'right_child' in root:
            getFeature(root['right_child'], res)
    getFeature(root, res)
    return res

def getTreeSplits(model):
    featurelist = []
    threhlist = []
    trees = []
    for idx, tree in enumerate(model['tree_info']):
        trees.append(TreeInterpreter(tree))
        featurelist.append(trees[-1].feature)
        threhlist.append(getItemByTree(trees[-1], 'threshold'))
    return (trees, featurelist, threhlist)


def getChildren(trees):
    listcl = []
    listcr = []
    for idx, tree in enumerate(trees):
        listcl.append(getItemByTree(tree, 'left_child'))
        listcr.append(getItemByTree(tree, 'right_child'))
    return(listcl, listcr)

class TreeInterpreter(object):
    def __init__(self, tree):
        self.raw = tree
        self.split_nodes = countSplitNodes(tree)
        self.node_count = self.split_nodes# + tree['num_leaves']
        self.value = getItemByTree(self, item='value')
        self.feature = getItemByTree(self)
        self.gain = getItemByTree(self, 'split_gain')
        # self.leaf_value = getLeafValue(tree)

class ModelInterpreter(object):
    def __init__(self, model, tree_model='lightgbm'):
        print("Model Interpreting...")
        self.tree_model = tree_model
        model = model.dump_model()
        self.n_features_ = model['max_feature_idx'] + 1
        self.trees, self.featurelist, self.threshlist = getTreeSplits(model)
        self.listcl, self.listcr = getChildren(self.trees)

    def GetTreeSplits(self):
        return (self.trees, self.featurelist, self.threshlist)

    def GetChildren(self):
        return (self.listcl, self.listcr)


    def EqualGroup(self, n_clusters, args):
        vectors = {}
        # n_feature = 256
        for idx,features in enumerate(self.featurelist):
            vectors[idx] = set(features[np.where(features>0)])
        keys = random.sample(vectors.keys(), len(vectors))
        clusterIdx = np.zeros(len(vectors))
        groups = [[] for i in range(n_clusters)]
        trees_per_cluster = len(vectors)//n_clusters
        mod_per_cluster = len(vectors) % n_clusters
        begin = 0
        for idx in range(n_clusters):
            for jdx in range(trees_per_cluster):
                clusterIdx[keys[begin]] = idx
                begin += 1
            if idx < mod_per_cluster:
                clusterIdx[keys[begin]] = idx
                begin += 1
        return clusterIdx

    def OptimizedGroups(self, n_clusters, args):

        clusterIdx = self.EqualGroup(n_clusters, args)

        cfs, cluster_sets, tree_sets = compute_group_sizes(n_clusters, clusterIdx.astype(np.int), self.featurelist)
        print(f"Features per group: {args.feat_per_group}")
        print("Cluster sets before:")
        print(cluster_sets)
        print(f"Missed features: {sum(max(0, f_len - args.feat_per_group) for _, f_len in cluster_sets)}")

        annealer = FeaturesAnnealer(n_clusters, clusterIdx, self.featurelist, args.feat_per_group)
        annealer.set_schedule(annealer.auto(minutes=1.0))
        clusterIdx, bestSum = annealer.anneal()

        cfs, cluster_sets, tree_sets = compute_group_sizes(n_clusters, clusterIdx.astype(np.int), self.featurelist)
        print("Cluster sets after:")
        print(cluster_sets)
        print(f"Missed features: {sum(max(0, f_len - args.feat_per_group) for _, f_len in cluster_sets)}")
        return clusterIdx


def compute_group_sizes(n_clusters, clusterIdx, featurelist):
    cluster_feature_sets = [set() for _ in range(n_clusters)]
    for cluster in range(n_clusters):
        # print(cluster)
        # print(clusterIdx)
        # print(np.where(clusterIdx == cluster)[0])
        for ind in np.where(clusterIdx == cluster)[0]:
            # print(ind)
            features = featurelist[ind]
            cluster_feature_sets[cluster] |= set(features[features > 0])

    cluster_sets = [(i, len(f_set)) for i, f_set in enumerate(cluster_feature_sets)]
    tree_sets = [(i, len(set(f_set[f_set > 0]))) for i, f_set in enumerate(featurelist)]

    return cluster_feature_sets, cluster_sets, tree_sets


class FeaturesAnnealer(Annealer):

    def __init__(self, n_clusters: int, clusterIdx: np.ndarray, featurelist: List[np.ndarray], feat_per_group: int):
        self.featurelist = featurelist
        self.n_clusters = n_clusters
        self.feat_per_group = feat_per_group
        state = clusterIdx.astype(np.int)
        super(FeaturesAnnealer, self).__init__(state)

    def move(self):
        while 1:
            i, j = np.random.randint(len(self.state), size=2)
            if self.state[i] != self.state[j]:
                self.state[i], self.state[j] = self.state[j], self.state[i]
                break

    def energy(self):
        _, cluster_sets, _ = compute_group_sizes(self.n_clusters, self.state, self.featurelist)
        return sum(max(0, f_len - self.feat_per_group) for _, f_len in cluster_sets)
