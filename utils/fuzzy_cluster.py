import torch
import torch.nn as nn
from utils.torch_utils import get_param_list
from utils.my_profiler import *
from sklearn.decomposition import PCA
import torch.nn.functional as F

from sklearn.decomposition import TruncatedSVD
import numpy as np
from scipy.spatial import distance


calc_time = True


class FuzzyCluster():

    def __init__(self, fuzzy_m=2.0):
        self.fuzzy_m = fuzzy_m

    def init_membership_mat(self, n_clients, n_clusters):

        membership_mat = torch.rand(n_clients, n_clusters)
        # 对每行进行归一化，使得每行的元素和为1
        membership_mat = membership_mat / \
            membership_mat.sum(dim=1, keepdim=True)

        print("init membership_mat: ", membership_mat[2:5])
        return membership_mat

    # @calc_exec_time(calc_time=True)
    # @memory_profiler
    def update_membership_mat(self, membership_mat, cluster_params, client_params, client_id):
        p = float(2 / (self.fuzzy_m - 1))
        n_clusters = cluster_params.size(0)

        distances = torch.zeros(n_clusters, device=client_params.device)
        with torch.no_grad():
            for i in range(n_clusters):
                diff = client_params - cluster_params[i]
                distances[i] = torch.norm(diff)
            print(distances)

            membership_mat[client_id] = F.softmax(distances.T).T
            print(membership_mat[client_id])

            # for cluster_id in range(n_clusters):
            #     den = 0.0
            #     for j in range(n_clusters):
            #         den += (distances[cluster_id] / distances[j])
            #     print("den ", den)
            #     membership_mat[client_id, cluster_id] = 1.0 / den
            #     torch.cuda.empty_cache()

        return membership_mat

    def update_membership_mahalanobis(self, membership_mat, cluster_params, client_params, client_id):

        p = float(2 / (self.fuzzy_m - 1))
        n_clusters = cluster_params.size(0)
        distances = torch.zeros(n_clusters, device=client_params.device)
        client_params = client_params.unsqueeze(0)
        # stack client_params and cluster_params
        X = torch.vstack([client_params, cluster_params])
        XT = X.T

        with torch.no_grad():
            # 1. compute covariance matrix and its inverse
            cov_matrix = torch.cov(X)
            print("cov_matrix.size", cov_matrix.size())
            inv_cov_matrix = torch.inverse(cov_matrix)
            print("inv_cov_matrix.size", inv_cov_matrix[0][0:5])

            for i in range(n_clusters):
                # 2. compute the difference between the client_params and each cluster_params
                diff = XT[0] - XT[i + 1]                    # XT[0] is client_params
                # 3. compute mahalanobis distance
                distances[i] = torch.sqrt((diff.T @ inv_cov_matrix @ diff))
                print("diff ", diff[i])

            print(distances)
            for cluster_id in range(n_clusters):
                den = 0.0
                for j in range(n_clusters):
                    den += (distances[cluster_id] / distances[j])**p
                print("den", den)
                membership_mat[client_id, cluster_id] = 1.0 / den
                torch.cuda.empty_cache()

        return membership_mat

    def update_membership_loss(self, membership_mat, losses, client_id):
        p = float(2 / (self.fuzzy_m - 1))
        n_clusters = losses.size(0)

        # 预先计算所有损失值
        with torch.no_grad():
            for cluster_id in range(n_clusters):
                den = 0.0
                for j in range(n_clusters):
                    # 使用损失值替代距离
                    den += (losses[cluster_id] / losses[j])**p
                    print("den ", den)
                membership_mat[client_id, cluster_id] = 1.0 / den.item()
                torch.cuda.empty_cache()

        return membership_mat

    def update_membership_normalize_maxmin(self, membership_mat, cluster_params, client_params, client_id):
        p = float(2 / (self.fuzzy_m - 1))
        n_clusters = cluster_params.size(0)

        # 对参数进行标准化处理，使其有0均值和1方差
        # client_params = (client_params - client_params.mean(dim=0, keepdim=True)) / client_params.std(dim=0, keepdim=True)
        # cluster_params = (cluster_params - cluster_params.mean(dim=1, keepdim=True)) / cluster_params.std(dim=1, keepdim=True)
        client_params = (client_params - client_params.mean(dim=0, keepdim=True))
        cluster_params = (cluster_params - cluster_params.mean(dim=1, keepdim=True))

        # 预先计算所有距离
        distances = torch.zeros(n_clusters, device=client_params.device)
        with torch.no_grad():
            for i in range(n_clusters):
                print("cluster_params[i] size", cluster_params[i][10:15])
                print("client_params ", client_params[10:15])
                diff = client_params - cluster_params[i]
                distances[i] = (torch.sum(diff * diff))                    # 直接将结果保存到 distances[i] 中
                distances[i].sqrt_()                    # 原地开方操作

                print(distances[i])

            for cluster_id in range(n_clusters):
                den = 0.0
                for j in range(n_clusters):
                    den += (distances[cluster_id] / distances[j])**p

                print("den ", den)
                membership_mat[client_id, cluster_id] = 1.0 / den
                torch.cuda.empty_cache()

        return membership_mat
# diff = (diff - diff.mean(dim=0, keepdim=True)) / diff.std(dim=0, keepdim=True)
# print("diff ",diff[10:20])

    @calc_exec_time(calc_time=True)
    def update_membership_normalize_std(self, membership_mat, cluster_params, client_params, client_id):
        p = float(1.5 / (self.fuzzy_m - 1))
        n_clusters = cluster_params.size(0)

        # 对参数进行标准化处理，使其有0均值和1方差
        # client_params = (client_params - client_params.mean(dim=0, keepdim=True)) / client_params.std(dim=0, keepdim=True)
        # cluster_params = (cluster_params - cluster_params.mean(dim=1, keepdim=True)) / cluster_params.std(dim=1, keepdim=True)
        # client_param_mean=client_params.mean(dim=0, keepdim=True)
        # client_param_std=client_params.std(dim=0, keepdim=True)
        # client_params = (client_params - clent_param_mean)/clent_param_std

        # distances = torch.cdist(client_params, cluster_params)
        # 预先计算所有距离
        distances = torch.zeros(n_clusters, device=client_params.device)
        with torch.no_grad():
            for i in range(n_clusters):
                print("cluster_params[i] ", cluster_params[i][10:15])
                print("client_params ", client_params[10:15])
                print("client_params size ", client_params.size())
                diff = client_params - cluster_params[i]
                print("diff ", diff[10:20])

                distances[i] = torch.norm(diff, dim=-1)
            print("distances ", distances)

            for cluster_id in range(n_clusters):
                den = 0.0
                for j in range(n_clusters):
                    den += (distances[cluster_id] / distances[j])**p

                print("den ", den)
                membership_mat[client_id, cluster_id] = 1.0 / den
                torch.cuda.empty_cache()

        return membership_mat

    def update_membership_cosine(self, membership_mat, cluster_params, client_params, client_id):
        p = float(2 / (self.fuzzy_m - 1))
        n_clusters = cluster_params.size(0)

        # 预先计算所有距离
        distances = torch.zeros(n_clusters, device=client_params.device)
        with torch.no_grad():
            for i in range(n_clusters):
                # 添加额外的维度以计算余弦相似度
                cos_sim = F.cosine_similarity(client_params.unsqueeze(0), cluster_params[i].unsqueeze(0))
                distances[i] = (1.0 - cos_sim)
            print("distances ", distances)

            for cluster_id in range(n_clusters):
                den = 0.0
                for j in range(n_clusters):
                    den += (distances[cluster_id] / distances[j])**p
                membership_mat[client_id, cluster_id] = 1.0 / den
                torch.cuda.empty_cache()

        return membership_mat

    def update_membership_mat2(self, membership_mat, cluster_params, client_params, client_id):
        p = float(2 / (self.fuzzy_m - 1))
        n_clusters = cluster_params.size(0)

        # 预先计算所有距离
        distances = torch.zeros(n_clusters, device=client_params.device)
        with torch.no_grad():
            for i in range(n_clusters):
                diff = client_params - cluster_params[i]
                distances[i].copy_(torch.sum(diff * diff))                    # 直接将结果保存到 distances[i] 中
                distances[i].sqrt_()                    # 原地开方操作

            for cluster_id in range(n_clusters):
                den = 0.0
                for j in range(n_clusters):
                    den += (distances[cluster_id] / distances[j])**p
                membership_mat[client_id, cluster_id] = 1.0 / den
                torch.cuda.empty_cache()

        return membership_mat

    def update_membership_pca(self, membership_mat, cluster_params, client_params, client_id):
        p = float(2 / (self.fuzzy_m - 1))
        n_clusters = cluster_params.size(0)

        # Convert tensors to numpy for PCA
        cluster_params_np = cluster_params.cpu().detach().numpy()
        client_params_np = client_params.cpu().numpy().reshape(1, -1)

        # Apply PCA
        pca = PCA(n_components=0.98)                    # 95% of variance explained
        cluster_params_pca = pca.fit_transform(cluster_params_np)
        client_params_pca = pca.transform(client_params_np)

        # Convert back to tensors
        cluster_params_pca = torch.from_numpy(cluster_params_pca).to(client_params.device)
        client_params_pca = torch.from_numpy(client_params_pca).to(client_params.device)
        # print("cluster_params_pca size", cluster_params_pca.size() )
        # print("cluster_params_pca ", cluster_params_pca )
        # print("client_params_pca size", client_params_pca.size() )
        # print("client_params_pca ", client_params_pca )
        distances = torch.zeros(n_clusters, device=client_params.device)
        with torch.no_grad():
            # for i in range(n_clusters):
            #     diff = client_params_pca - cluster_params_pca[i]
            #     distances[i] = (torch.sum(diff * diff)).sqrt()  # directly save the result into distances[i]
            #     print(distances[i])

            # for i in range(n_clusters):
            #     diff = client_params_pca - cluster_params_pca[i]
            #     distances[i] = torch.norm(diff)  # 计算欧氏距离并保存结果到distances[i]
            #     print(distances[i])
            distances = torch.norm(client_params_pca - cluster_params_pca, dim=1)
            for cluster_id in range(n_clusters):
                den = 0.0
                for j in range(n_clusters):
                    den += (distances[cluster_id] / distances[j])**p
                print("den ", den)
                membership_mat[client_id, cluster_id] = 1.0 / den
                torch.cuda.empty_cache()

        return membership_mat

    def update_membership_edc(self, membership_mat, cluster_params, client_params, client_id):
        p = float(2 / (self.fuzzy_m - 1))
        n_clusters = cluster_params.size(0)

        distances = torch.zeros(n_clusters, device=client_params.device)
        with torch.no_grad():
            # 客户端参数转换为 numpy
            client_params_np = client_params.cpu().numpy().reshape(1, -1)

            for i in range(n_clusters):
                # 单个蔟参数转换为 numpy
                cluster_param_i_np = cluster_params[i].cpu().numpy().reshape(1, -1)

                # SVD
                svd = TruncatedSVD(n_components=n_clusters)
                decomp_updates = svd.fit_transform(cluster_param_i_np.T)

                # 计算 cosine similarity
                decomposed_cossim_matrix = F.cosine_similarity(client_params_np, decomp_updates.T)

                # 获取对应的 EDC
                distances[i] = torch.tensor(decomposed_cossim_matrix[0, 0])

            for cluster_id in range(n_clusters):
                den = 0.0
                for j in range(n_clusters):
                    den += (distances[cluster_id] / distances[j])**p
                membership_mat[client_id, cluster_id] = 1.0 / den
                torch.cuda.empty_cache()

        return membership_mat

    @memory_profiler
    @calc_exec_time(calc_time=calc_time)
    def update_membership_mat1(self, membership_mat, cluster_params, client_params, client_id):
        p = float(2 / (self.fuzzy_m - 1))

        n_clusters = cluster_params.size(0)

        distances = torch.cdist(client_params, cluster_params)
        for cluster_id in range(n_clusters):
            den = torch.sum((distances[0, cluster_id] / distances[0, :])**p)
            membership_mat[client_id, cluster_id] = 1.0 / den

        return membership_mat

    def get_clusters(self, membership_mat):
        cluster_labels = torch.argmax(membership_mat, dim=0)
        return cluster_labels.tolist()


# # Example usage:
# # Assuming `data_points` is a list of torch tensors representing the data points
# fuzzy_cluster = FuzzyCluster()
# cluster_labels, cluster_centers = fuzzy_cluster.fuzzyCMeansClustering(data_points, n_cluster)
