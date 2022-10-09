# SOMAD

## 运行
python somad.py --dataset mvtec

## 默认参数说明

1.默认数据集 mvtec
2.默认数据集存放地址：./mvtec存放解压后的数据集
eg:
./mvtec/bottle
....
3.默认结果保存结果根目录 ./result
4.默认top_k 4
5.默认聚类数 3136
6.聚类方法,通过cluster_device参数来设置：
- cpu基于numpy的sompy库(对初始化部分有修改，具体sompy/sompy.py和codebook.py)
- gpu 基于pytorch的kmeans算法（初始化也有修改，kmeans.py）
7.目前最好的score计算方式： mahalanobis

## 保存结果文件夹组织
./result
	mvtec
		gts
		 {类别}.pkl	ground truth保存成未见
		method
			network
				dimensions
					clusters 聚类结果
						cpu聚类方法
							类别+聚类数.pkl
						gpu
					evaluation
						{聚类数}centers_top{k}_{score_type}
							visualize可视化
							result.pkl 评测中间结果的数据
							roc_auc.png roc_auc图
					feats 实际未保存特征。
	dagm

