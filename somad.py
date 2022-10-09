import random
from random import sample
import argparse
import numpy as np
import os
import pickle
import joblib
from tqdm import tqdm
from utils.decorators import timeit
from multiprocessing.dummy import Pool as ThreadPool
from collections import OrderedDict
from scipy.spatial.distance import mahalanobis


from kmeans import cluster
from scipy.ndimage import gaussian_filter

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from model import wide_resnet50_2
import datasets.mvtec as mvtec
import datasets.dagm as dagm
from evaluate.evaluation import Evaluator

#region+ device setup
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
random.seed(1024)
torch.manual_seed(1024)
if use_cuda:
    torch.cuda.manual_seed_all(1024)
#endregion

def parse_args():
    parser = argparse.ArgumentParser('PaDiM')
    parser.add_argument("--dataset", type=str, choices=['mvtec', 'dagm'], default="mvtec")
    parser.add_argument("--data_path", type=str, default="./mvtec")
    parser.add_argument("--save_path", type=str, default="result/")
    parser.add_argument("--method",type=str,default='som_2d')
    parser.add_argument("--arch", type=str, choices=['resnet18, wide_resnet50_2'], default='wide_resnet50_2')
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--n_clusters",type=int,default=3136)
    parser.add_argument("--score_type",type=str,choices=['knn','center','mahalanobis'],default='mahalanobis')
    parser.add_argument("--cluster_device",type=str,default='cpu')
    parser.add_argument("--experiment_name",type=str,default='')
    return parser.parse_args()


class SOM_AD:
    @timeit('SOM_AD initialize')
    def __init__(self,args):
        self._update_args(args)
        self.equal=False
        self._select_datset()
        self._init_model(550)
        self._add_hook()
        self._init_path()

        self.evaluator = Evaluator(args)
        self.tested_class = list(self.evaluator.thresholds.keys())
        if len(self.tested_class)>0:
            self.evaluator.load_result()
        else:
            print('start from empty')

    #region+ method called by init
    def _update_args(self,args):
        self.base_save_dir=args.save_path
        args.save_path = os.path.join(args.save_path, args.dataset,args.method)
        args.data_path = './' + args.dataset
        args.experiment_name='%dcenters_top%d_%s'%(args.n_clusters,args.top_k,args.score_type)+args.experiment_name
        self.args = args

    def _select_datset(self):
        if self.args.dataset == 'mvtec':
            self.dataset = mvtec
        else:
            self.dataset = dagm

    def _init_model(self,selected_dimensions=550):
        self.use_all=False
        if self.args.arch == 'resnet18':
            self.model = resnet18(pretrained=True, progress=True)
            self.t_d = 448
            self.d = selected_dimensions
        elif self.args.arch == 'wide_resnet50_2':
            self.model = wide_resnet50_2(pretrained=True, progress=True)
            self.t_d = 1792
            self.d = selected_dimensions
        self.model.to(device)
        self.model.eval()
        if self.t_d>self.d:
            if self.equal:
                selected_layer1=sample(range(0,256),128)
                selected_layer2=sample(range(256,768),128)
                selected_layer3=sample(range(768,1792),768)
                self.selected_indexes = torch.tensor(selected_layer1 + selected_layer2 + selected_layer3)
            else:
                self.selected_indexes = torch.tensor(sample(range(0, self.t_d), self.d))


        elif self.t_d==self.d:
            self.use_all=True
        else:
            raise Exception('[selected_dimensions] out of range,max dimensions is %d'%self.t_d)

    def _add_hook(self):
        self.outputs = []
        def hook(module, input, output):
            self.outputs.append(output)
        self.model.layer1[-1].register_forward_hook(hook)
        self.model.layer2[-1].register_forward_hook(hook)
        self.model.layer3[-1].register_forward_hook(hook)

    def _init_path(self):
        self.save_path = os.path.join(self.args.save_path, self.args.arch, '%d_dimensions' % self.d)
        # only related with dataset eg: res/mvetec/gts
        self.gt_path = os.path.join(self.base_save_dir,self.args.dataset, 'gts')
        os.makedirs(self.gt_path, exist_ok=True)

        self.args.save_path = self.save_path
        feat_path = os.path.join(self.save_path, 'feats')
        os.makedirs(feat_path, exist_ok=True)
        self.cluster_path = os.path.join(self.save_path, 'clusters', self.args.cluster_device)
        os.makedirs(self.cluster_path, exist_ok=True)

    #endregion

    def _init_dataset(self,class_name):
        train_dataset = self.dataset.MVTecDataset(self.args.data_path, class_name=class_name, is_train=True)
        self.train_dataloader=DataLoader(train_dataset, batch_size=32, pin_memory=True)
        test_dataset = self.dataset.MVTecDataset(self.args.data_path, class_name=class_name, is_train=False)
        self.test_dataloader = DataLoader(test_dataset, batch_size=32, pin_memory=True)

    @timeit('Load Ground Truth')
    def _get_GTs(self,class_name):
        gt_file_path=os.path.join(self.gt_path,'%s.pkl'%class_name)
        if not os.path.isfile(gt_file_path):
            gt_list = []
            gt_mask_list = []
            test_imgs = []
            for (x, y, mask) in tqdm(self.test_dataloader, '| load ground Truth | test | %s |' % class_name):
                test_imgs.extend(x.cpu().detach().numpy())
                gt_list.extend(y.cpu().detach().numpy())
                gt_mask_list.extend(mask.cpu().detach().numpy())
            with open(gt_file_path,'wb') as f:
                data={}
                data['test_imgs']=test_imgs
                data['gt_list']=gt_list
                data['gt_mask_list']=gt_mask_list
                joblib.dump(data,f)
        else:
            with open(gt_file_path, 'rb') as f:
                data=joblib.load(f)
                test_imgs=data['test_imgs']
                gt_list=data['gt_list']
                gt_mask_list=data['gt_mask_list']
        return gt_list,gt_mask_list,test_imgs

    def _extract_features(self,class_name,mode):
        if mode=='train':
            dataloader = self.train_dataloader
        else:
            dataloader = self.test_dataloader
        temp_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
        for (x, _, _) in tqdm(dataloader, '| feature extraction | %s | %s |' %(mode,class_name)):
            # model prediction
            with torch.no_grad():
                _ = self.model(x.to(device))
            # get intermediate layer outputs
            for k, v in zip(temp_outputs.keys(), self.outputs):
                if k != 'layer1':
                    v = F.interpolate(v, size=56, mode='bilinear', align_corners=True)
                temp_outputs[k].append(v.cpu().detach())
            # initialize hook outputs
            self.outputs = []
        for k, v in temp_outputs.items():
            temp_outputs[k] = torch.cat(v, 0)
        embedding_vectors = torch.cat(
            [temp_outputs['layer1'], temp_outputs['layer2'], temp_outputs['layer3']],
            dim=1)
        # randomly select d dimension
        if not self.use_all:
            embedding_vectors = torch.index_select(embedding_vectors, 1, self.selected_indexes)
        return embedding_vectors

    @timeit('Clustering')
    def _cluster(self,class_name,embeddings_vectors):
        if self.args.cluster_device == 'cpu':
            train_embeddings = embeddings_vectors.numpy()
        else:
            train_embeddings = embeddings_vectors.cuda()

        cluster_path_ = os.path.join(self.cluster_path, '%s_%d.pkl' % (class_name, self.args.n_clusters))
        if not os.path.exists(cluster_path_):
            print('starting clustering %d' % self.args.n_clusters)
            if self.args.cluster_device == 'cpu':
                from sompy import SOMFactory
                smap = SOMFactory().build(train_embeddings, mapsize=(56,56), normalization=None,
                                          initialization='average')
                smap.train(n_job=4, verbose='info', train_rough_len=20, train_finetune_len=15)
                cluster_labels, cluster_centers = smap._bmu, smap.codebook.matrix
                with open(cluster_path_, 'wb') as f:
                    joblib.dump({'centers': cluster_centers, 'labels': cluster_labels}, f)
                print('%s saved!' % class_name)
            else:
                cluster_centers, cluster_labels = cluster(train_embeddings, self.args.n_clusters, self.args.n_clusters)
                cluster_centers=cluster_centers.cpu().detach().numpy()
                cluster_labels=cluster_labels.cpu().detach().numpy()
                with open(cluster_path_, 'wb') as f:
                    pickle.dump({'centers':cluster_centers,
                                 'labels': cluster_labels}, f)
            print('saving clustering %d result' % self.args.n_clusters)
        else:
            print('loading clustering %d' % self.args.n_clusters)
            with open(cluster_path_, 'rb') as f:
                data = joblib.load(f)
            cluster_centers = data['centers']
            cluster_labels = data['labels']
        return cluster_centers,cluster_labels

    def _compute_anomaly_score_mahalanobis(self,train_embeddings,test_embeddings,cluster_centers,cluster_labels):
        cluster_centers = torch.from_numpy(cluster_centers).cuda()
        labels=cluster_labels[0]
        unique_labels=np.unique(labels)

        I = np.identity(550)
        train_distribution=[]
        for label in unique_labels:
            temp_embeddings=train_embeddings[cluster_labels[0]==label].numpy()
            mean=temp_embeddings.mean(axis=0)
            cov=np.cov(temp_embeddings, rowvar=False) + 0.01 * I
            train_distribution.append((mean,np.linalg.inv(cov)))
        dist_matrix = [0] * test_embeddings.shape[0]

        vectors_per_loop=50000
        for batch in tqdm(range(int(np.ceil(test_embeddings.shape[0] / vectors_per_loop))),'batch vectors'):
            dist_centers=[]
            batch_temp=test_embeddings[batch*vectors_per_loop:(batch+1)*vectors_per_loop].cuda()
            for i in range(cluster_centers.shape[0]):
                temp=torch.pairwise_distance(batch_temp,cluster_centers[i].unsqueeze(dim=0)).unsqueeze(dim=0)
                dist_centers.append(temp)
            dist_centers=torch.cat(dist_centers,dim=0).transpose(0,1)
            # 50000*3136
            _, min_center_indexes = torch.topk(dist_centers, k=self.args.top_k, dim=1, largest=False)
            batch_temp=batch_temp.detach().cpu().numpy()

            def process_vector(param):
                idx,min_center_index=param
                min_center_index = [item.item() for item in min_center_index if item.item() in cluster_labels[0]]
                if len(min_center_index) != 0:
                    dists=[]
                    for index in min_center_index:
                        n_idx=np.argwhere(unique_labels==index)[0][0]
                        dists.append(mahalanobis(batch_temp[idx],train_distribution[n_idx][0],train_distribution[n_idx][1]))
                    dist_matrix[batch*vectors_per_loop+idx]=min(dists)
                else:
                    dist_matrix[batch*vectors_per_loop+idx]=-1
            pool = ThreadPool(10)
            pool.map(process_vector, enumerate(min_center_indexes))
            pool.close()
            pool.join()

            # for idx,min_center_index in enumerate(min_center_indexes):
            #     min_center_index = [item.item() for item in min_center_index if item.item() in cluster_labels[0]]
            #     if len(min_center_index) != 0:
            #         nearest_feats = torch.cat(
            #             [train_embeddings[index == cluster_labels[0]] for index in min_center_index], dim=0)
            #         dists = torch.pairwise_distance(nearest_feats,batch_temp[idx].unsqueeze(dim=0))
            #         dist_matrix[batch*vectors_per_loop+idx]=dists.min()
            #     else:
            #         dist_matrix[batch*vectors_per_loop+idx]=-1
            print('done batch %d' % batch)
        dist_matrix=np.array(dist_matrix)
        dist_matrix[dist_matrix==-1]=dist_matrix.max()
        return dist_matrix

    def _compute_anomaly_score_knn(self,train_embeddings,test_embeddings,cluster_centers,cluster_labels):
        dist_matrix = [0]*test_embeddings.shape[0]
        train_embeddings=train_embeddings.cuda()
        # test_embeddings=test_embeddings.cuda()
        cluster_centers=torch.from_numpy(cluster_centers).cuda()
        vectors_per_loop=50000
        for batch in tqdm(range(int(np.ceil(test_embeddings.shape[0] / vectors_per_loop))),'batch vectors'):
            dist_centers=[]
            batch_temp=test_embeddings[batch*vectors_per_loop:(batch+1)*vectors_per_loop].cuda()
            for i in range(cluster_centers.shape[0]):
                temp=torch.pairwise_distance(batch_temp,cluster_centers[i].unsqueeze(dim=0)).unsqueeze(dim=0)
                dist_centers.append(temp)
            dist_centers=torch.cat(dist_centers,dim=0).transpose(0,1)
            # 50000*3136
            _, min_center_indexes = torch.topk(dist_centers, k=self.args.top_k, dim=1, largest=False)

            def process_vector(param):
                idx,min_center_index=param
                min_center_index = [item.item() for item in min_center_index if item.item() in cluster_labels]
                if len(min_center_index) != 0:
                    nearest_feats = torch.cat(
                        [train_embeddings[index == cluster_labels[0]] for index in min_center_index], dim=0)
                    dists = torch.pairwise_distance(nearest_feats,
                                                    batch_temp[idx].unsqueeze(dim=0))
                    dist_matrix[batch*vectors_per_loop+idx]=dists.min().item()
                else:
                    dist_matrix[batch*vectors_per_loop+idx]=-1
            pool = ThreadPool(10)
            pool.map(process_vector, enumerate(min_center_indexes))
            pool.close()
            pool.join()

            # for idx,min_center_index in enumerate(min_center_indexes):
            #     min_center_index = [item.item() for item in min_center_index if item.item() in cluster_labels[0]]
            #     if len(min_center_index) != 0:
            #         nearest_feats = torch.cat(
            #             [train_embeddings[index == cluster_labels[0]] for index in min_center_index], dim=0)
            #         dists = torch.pairwise_distance(nearest_feats,batch_temp[idx].unsqueeze(dim=0))
            #         dist_matrix[batch*vectors_per_loop+idx]=dists.min()
            #     else:
            #         dist_matrix[batch*vectors_per_loop+idx]=-1
            print('done batch %d' % batch)
        dist_matrix=np.array(dist_matrix)
        dist_matrix[dist_matrix==-1]=dist_matrix.max()
        return dist_matrix

    def _compute_anomaly_score_center(self,test_embeddings, cluster_centers, cluster_labels):
        dist_matrix = [0] * test_embeddings.shape[0]
        test_embeddings=test_embeddings.cuda()
        cluster_centers = torch.from_numpy(cluster_centers).cuda()
        vectors_per_loop = 50000
        labels=np.unique(cluster_labels[0])
        for batch in tqdm(range(int(np.ceil(test_embeddings.shape[0] / vectors_per_loop))), 'batch vectors'):
            dist_centers = []
            batch_temp = test_embeddings[batch * vectors_per_loop:(batch + 1) * vectors_per_loop].cuda()
            for i in range(cluster_centers.shape[0]):
                temp = torch.pairwise_distance(batch_temp, cluster_centers[i].unsqueeze(dim=0)).unsqueeze(dim=0)
                dist_centers.append(temp)
            dist_centers = torch.cat(dist_centers, dim=0).transpose(0, 1)
            # 50000*3136
            dist, min_center_indexes = torch.topk(dist_centers, k=self.args.top_k, dim=1, largest=False)
            dist=dist.cpu().detach().numpy()
            min_center_indexes=min_center_indexes.cpu().detach().numpy()
            def process_vector(param):
                idx,(dist,min_center_index) = param
                near_dists = [dist for idx,(dist,item) in enumerate(zip(dist,min_center_index)) if item in labels]
                if len(near_dists) != 0:
                    dist_matrix[batch * vectors_per_loop + idx] = np.array(near_dists).mean()
                else:
                    dist_matrix[batch * vectors_per_loop + idx] = -1
            pool = ThreadPool(10)
            pool.map(process_vector, enumerate(zip(dist,min_center_indexes)))
            pool.close()
            pool.join()
            print('done batch %d' % batch)
        dist_matrix = np.array(dist_matrix)
        dist_matrix[dist_matrix == -1] = dist_matrix.max()
        return dist_matrix

    @timeit('Compuate Anomaly Score')
    def _compute_anomaly_score(self,train_embeddings,test_embeddings,cluster_centers,cluster_labels,method='center'):
        if method=='knn':
            return self._compute_anomaly_score_knn(train_embeddings,test_embeddings,cluster_centers,cluster_labels)
        elif method=='center':
            return self._compute_anomaly_score_center(test_embeddings,cluster_centers,cluster_labels)
        elif method=='mahalanobis':
            return self._compute_anomaly_score_mahalanobis(train_embeddings,test_embeddings,cluster_centers,cluster_labels)

    @timeit('Anomaly Map Postprocess')
    def _post_processing(self,dist_matrix,result_size=224,sigma=4):
        score_map_list = []
        for i in range(dist_matrix.shape[0]):
            score_map = F.interpolate(dist_matrix[i].unsqueeze(0).unsqueeze(0), size=result_size,
                                  mode='bilinear', align_corners=False)
            # apply gaussian smoothing on the score map
            score_map = gaussian_filter(score_map.squeeze().cpu().detach().numpy(), sigma=sigma)
            score_map_list.append(score_map)
        score_map_list=np.array(score_map_list)
        # Normalization
        max_score = score_map_list.max()
        min_score = score_map_list.min()
        pixel_scores = (score_map_list - min_score) / (max_score - min_score)
        img_scores = pixel_scores.reshape(pixel_scores.shape[0], -1).max(axis=1)
        return img_scores,pixel_scores

    @timeit('Evalute')
    def _evalute(self,class_name,img_scores,pixel_scores):
        gt_list, gt_mask_list, test_imgs = self._get_GTs(class_name)
        self.evaluator.eval_class(class_name, gt_list, gt_mask_list, img_scores, pixel_scores)
        self.evaluator.visualize_class(class_name, test_imgs, pixel_scores, gt_mask_list)

    @timeit('Process cate')
    def process_class(self, class_name,cluster_only=False):
        self._init_dataset(class_name)
        self.model = self.model.cuda()
        train_embeddings = self._extract_features(class_name, 'train').transpose(1, 3).flatten(0, 2)
        if not cluster_only:
            test_embeddings = self._extract_features(class_name, 'test')
        # self._get_GTs(class_name)
        self.model = self.model.cpu()
        torch.cuda.empty_cache()

        cluster_centers, cluster_labels = self._cluster(class_name, train_embeddings)
        if not cluster_only:
            B, C, H, W = test_embeddings.shape
            test_embeddings = test_embeddings.transpose(1, 3).flatten(0, 2)
            dist_matrix = self._compute_anomaly_score(train_embeddings, test_embeddings, cluster_centers, cluster_labels,self.args.score_type)
            dist_matrix = torch.from_numpy(dist_matrix).cuda()
            dist_matrix = dist_matrix.view(B, H, W).transpose(1, 2)
            img_scores, pixel_scores = self._post_processing(dist_matrix)
            self._evalute(class_name, img_scores, pixel_scores)

    # @timeit('Process All')
    # def process_all(self):
    #     for class_name in self.dataset.CLASS_NAMES:
    #         if class_name in self.tested_class:
    #             continue
    #             # ['grid','metal_nut','transistor','zipper','capsule']
    #         if class_name not in ['bottle']:
    #             continue
    #         try:
    #             self.process_class(class_name,False)
    #         except Exception as e:
    #             print('%s abort' % class_name)
    #             print(e.args)
    #         continue
    #     self.evaluator.save()

    def process_all(self):
        for class_name in self.dataset.CLASS_NAMES:
            if class_name in self.tested_class:
                continue
            # if class_name!="screw":
            #     continue
            self.process_class(class_name,False)
        self.evaluator.save()


def main():
    args = parse_args()
    somad=SOM_AD(args)
    somad.process_all()


def calc_dist_matrix(x, y):
    """Calculate Euclidean distance matrix with torch.tensor"""
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    dist_matrix = torch.sqrt(torch.pow(x - y, 2).sum(2))
    return dist_matrix


def embedding_concat(x, y):
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)
    return z


if __name__ == '__main__':
    main()