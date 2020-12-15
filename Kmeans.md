## K-Means聚类————基于距离的算法（十大经典算法之一）
K-Means 聚类采用距离作为相似性的评价指标，即认为两个对象的距离越近，其相似度就越大。
### 一. 原理步骤：  
这是一个循环迭代的算法，简单易懂。假定我们要对 N 个样本观测做聚类：  
(1) 假设聚为 K 类，并且随机选择 K 个点作为初始中心点；  
(2) 接下来，按照距离初始中心点最小的原则，把所有观测分到各中心点所在的类中；  
(3) 每类中有若干个观测，计算 K 个类中所有样本点的均值，作为第二次迭代的 K 个中心点；  
(4) 然后根据这个中心重复第 2、3 步，直到收敛（中心点不再改变或达到指定的迭代次数），聚类过程结束。  
下图展示最简单的聚类，分 2 类：  
![](https://ftp.bmp.ovh/imgs/2020/12/12cb6a37de432746.png)  
实际情况中，质心一般要迭代多次，才能达到相对最优情况。
>
### 二. 算法优缺点：
#### (2.1) 优点：  
- 该算法时间复杂度为O(tkmn)，（其中，t为迭代次数，k为簇的数目，m为记录数，n为维数）与样本数量线性相关，所以，对于处理大数据集合，该算法非常高效    
- 原理简单，较易理解    
>
#### (2.2) 缺点：  
- 易受异常值和噪声影响 （异常值使均值偏离）   
- 结果不一定是全局最优，只能保证局部最优（和初始点选取有关）
- 聚类中心的个数 K 需要事先给定，但在实际应用中这个 K 值的选定是非常难以估计的
- 不适于发现非凸面形状的簇或大小差别很大的簇（不均衡样本）
- k 个初始化的质心的位置选择对最后的聚类结果和运行时间都有很大的影响,质心如果太近会影响聚类效果。
>
#### (2.3) 改善方法：  
- 初始选择质心问题：针对最初随机选取 k 个质心可能使收敛变慢、模型效果不佳问题，因此推出 K-Means++ 算法，其优化了初始质心的选择，更合理去选择，从而优化模型。scikit-learn 默认使用 K-Means++ 算法，也可调回 random ，变回最原始的算法。
- 计算速度优化——1：传统的K-Means需要不断计算所有点到质心的距离，新的elkan K-Means利用三角形性质：两边之和大于等于第三边,以及两边之差小于第三边，来减少距离的计算，有效提高迭代速度。针对稠密数据可用，若是稀疏数据，有缺失值，该方法行不通，只能用回原来的距离计算方法。[详见大神博客](https://www.cnblogs.com/pinard/p/6164214.html)
- 计算速度优化——2：样本量很大的话，也需要消耗较长计算时间，比如样本量达到10万、特征有100以上，可以考虑用Mini Batch K-Means。它以抽样的方式选出样本再计算，可以减少计算量，提高迭代速度。这会牺牲掉一定的精度，为了增加算法的准确性，一般会多跑几次，用得到不同的随机采样集来得到聚类簇，选择其中最优的聚类簇。
>
### 三. k值确定：
#### (3.1) 个人需求/经验  
比如，做客户分层，你想分成4部分，那么k值就取你想要的4。
>
#### (3.2) 手肘法Elbow method——看图辨别拐点   
尝试不同的K值，并且计算对应的集合内误差平方和:Within Set Sum of Squared Error(WSSSE)或者SSE(sum of the squared errors，误差平方和)，都是一样的。
![](https://ftp.bmp.ovh/imgs/2020/12/d8205cecb29c6e47.png)  
- (3.2.1) SSE参数解释：  
> Ci是第i个簇  
> p是Ci中的样本点  
> mi是Ci的质心（Ci中所有样本的均值）  
> SSE是所有样本的聚类误差，代表了聚类效果的好坏。  
- (3.2.2) SSE图解释：  
![](https://ftp.bmp.ovh/imgs/2020/12/8a110f66530ac3d8.png)
![](https://ftp.bmp.ovh/imgs/2020/12/a14c3c4c783d1f20.png)
SSE=所有的蓝色的线的平方加起来。假设样本数量为n，当k=1，SSE最大；当k=n，SSE达到最小，SSE=0。
SSE随着聚类数目增多而不断减小，并且SSE会由变化很快到最后平缓下来，当SSE减少得很缓慢时，就认为进一步增大聚类数效果也聚类效果也没有太明显的变化。关注斜率最大处，一个明显的“肘点”就是最佳聚类数目。
- (3.2.3) 举例SSE图表示：
![](https://pic2.zhimg.com/v2-25b396108e9b5da6094c2097888f2251_b.png)   
此时，选k=3。  
>
#### (3.3) Gap statistic   
上述手肘法Elbow method需要人眼去观察最佳值，有时并不能确切看出哪个值是真正的拐点，后来有斯坦福大佬推出更智能的Gap statistic方法。只需要找出使Gap Statistic最大的K值即可。
[安装包位置](https://www.cnpython.com/pypi/gapkmean) 用不了，不知道为啥，还是要自己造轮子才行
>
#### (3.4) 轮廓系数Silhouette Coefficient
以簇内的稠密程度和簇间的离散程度来评估聚类的效果，选择使轮廓系数较大对应的k值。用法参考：sklearn.metrics.silhouette_score
![](https://ftp.bmp.ovh/imgs/2020/12/5d68e3b232c1ac2e.png)  
其中, a(i):样本Xi到所有它属于的簇中其它点的距离的平均； b(i):样本Xi到所有非本身所在簇中其它点的距离的平均距离
轮廓系数取值范围在\[-1,1]之间。该值越大，越合理
- S(i) 接近 1, 说明样本 i 聚类合理
- S(i) 接近 -1, 则说明样本 i 更应该分类到另外的簇
- 若 s(i) 近似为 0, 则说明样本 i 在两个簇的边界上
用法参考：sklearn.metrics.silhouette_score
>
#### (3.5) Calinski-Harabasz Index 即(CH)指标  
同样也是以簇内的稠密程度和簇间的离散程度来评估聚类的效果，选择使Calinski-Harabasz分数较大对应的k值。其计算速度快很多。
公式：![](https://ftp.bmp.ovh/imgs/2020/12/692a7a67f082412e.png)  
其中，m为训练集样本数，k为类别数。Bk为类别之间的协方差矩阵，Wk为类别内部数据的协方差矩阵。tr为矩阵的迹。
类别内部数据的协方差越小越好，类别之间的协方差越大越好，这样的Calinski-Harabasz分数会高。  
[用法参考](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)sklearn.metrics.calinski_harabaz_score

>
### 四. 举栗子：
下面用sklearn的make_blobs方法来生成聚类算法的测试数据：  
先生成模拟数据：  
```
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @FileName: github_kmeans.py
# @Software: PyCharm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
from sklearn.datasets import make_blobs  # 生成聚类数据
from sklearn.cluster import KMeans
from sklearn import metrics

X, Y = make_blobs(n_samples=10000,  # 设置样本量
                  n_features=2,  # 设置样本特征个数
                  centers=[[-2, -3], [0, 2], [6, 0], [5, 6]],  # 设置簇的中心，从而也知道了簇/类的个数
                  cluster_std=[0.5, 0.4, 0.6, 0.6],  # 设置簇的方差
                  random_state=20)
plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y)
plt.grid()
plt.title('模拟数据')
plt.show()
```
![](https://ftp.bmp.ovh/imgs/2020/12/d9ddc76ba2990827.png)  
>
#### (4.1) 手肘法
```
# 1.手肘法
SSE = []
for k in range(1, 10):
    model = KMeans(n_clusters=k)  # 构造聚类器
    model.fit(X)
    SSE.append(model.inertia_)  # estimator.inertia_获取聚类准则的总和
x_1 = range(1, 10)
plt.xlabel('k值')
plt.ylabel('SSE')
plt.plot(x_1, SSE, 'o-')
plt.title('手肘法——寻找拐点')
plt.show()
```
![](https://ftp.bmp.ovh/imgs/2020/12/b596387822666121.png)  
>
#### (4.2) Gap statistic
```
# 2.Gap statistic
# # 后续补
```
>
#### (4.3) 轮廓系数Silhouette Coefficient
```
# 3.轮廓系数Silhouette Coefficient
Silhouette_Coeff = []
for k in range(2, 10):
    model_3 = KMeans(n_clusters=k, random_state=666).fit(X)
    y3_pred = model_3.labels_
    sh_score = metrics.silhouette_score(X, y3_pred, metric='euclidean')
    Silhouette_Coeff.append(sh_score)
# print(Silhouette_Coeff)
plt.plot(range(2, 10), Silhouette_Coeff, 'o-')
plt.title('轮廓系数——寻找最接近1的点')
plt.show()
```
![](https://ftp.bmp.ovh/imgs/2020/12/c149b19d6a602f12.png)  
>
#### (4.4) Calinski-Harabasz Index，(CH)指标
```
# 4.Calinski-Harabasz Index 即(CH)指标
CH_Score = []
for i in range(1, 10):
    y4_pred = KMeans(n_clusters=i+1, random_state=666).fit_predict(X)
    # plt.subplot(3,3,i)
    # plt.scatter(X[:, 0], X[:, 1], c=y4_pred)
    # plt.suptitle('k值:2-10')
    # plt.show()
    CH_Score.append(metrics.calinski_harabasz_score(X, y4_pred))
print('CH分数:\n', CH_Score)
plt.plot(range(2,11), CH_Score, 'o-')
plt.title('CH指标——寻找最大值')
plt.show()
```
![](https://ftp.bmp.ovh/imgs/2020/12/09f9cedd34d37156.png)  
>

### 五. 总结：
#### (5.1) 本文选取K值的评判标准仅是其中几个，后面可深挖不同指标的类别
#### (5.2) Kmeans聚类的结果因初始质心的选择问题，避免不了局部最优，实际中多聚类几次，用最好的结果更可靠
#### (5.3) Gap statistic的实现方法还没整理出来，(CH)指标的计算速度相对轮廓系数快，数据量很大时择(CH)指标更合适

如有错误，欢迎指错，谢谢各位大佬  
![](https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1608021441542&di=3e0b00c230f1eac1f1be61e632298caf&imgtype=0&src=http%3A%2F%2Fsearchfoto.ru%2Fimg%2FxyygpKbDS1y8pTjXUy83VS8rMS9fLSy3RL8nQz0zR9_cM0AtLNfH0dQ9JqUgx8NXNz7KwTClzSy93BAL7clsjYwO1xNwC6wxbQ3MIq6jY1hDMKEjOsU0BAzA339YUIpyZYmuoZwgA.jpg)
