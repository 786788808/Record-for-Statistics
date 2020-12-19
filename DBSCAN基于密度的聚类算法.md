## 基于密度的聚类算法DBSCAN(Density-Based Spatial Clustering of Applications with Noise)
本文介绍基于密度的聚类算法，这类算法假定聚类结构能通过样本分布的紧密程度确定。  
通常情况下，密度聚类算法从样本密度的角度来考察样本之间的可连接性，并给予可连接样本不断扩展聚类簇以获得最终的聚类结果。    
下面讲解其中一种方法：DBSCAN  
>
### 一. DBSCAN基础概念及原理步骤：  
DBSCAN(具有噪声的基于密度的聚类方法)是其中一种基于密度的聚类算法,  它基于一组邻域参数(ϵ, MinPts)来刻画样本分布的紧密程度。(ϵ读作 epsilon，有人缩写成 eps)
#### (1.1) 基本概念：  
给定数据集D={x1, x2, x3, ……, xm},
- ϵ-邻域：对于 xj∈D，其 ϵ-邻域包含样本集 D 中与 xj 的距离不大于 ϵ 的子样本集，即Nϵ(xj)={xi∈D|dist(xi,xj)≤ϵ}。定义点 xj 领域里的其他其他点的个数记为|Nϵ(xj)|　(注：默认选用欧式距离来作为dist()函数)   
- 核心对象：对于任一样本 xj∈D，如果其 ϵ-邻域 对应的 Nϵ(xj) 至少包含 MinPts 个样本，即如果|Nϵ(xj)|≥MinPts，则 xj 是核心对象。(可以将ϵ理解为半径，MinPts 理解为点周围的点个数(理论值)。以某个点为中心画个圆，如果圈里面的数量达标，它就是核心对象)  
- 密度直达：如果 xj 位于 xi 的 ϵ-邻域中，且 xi 是核心对象，则称 xj 由 xi 密度直达。(注意位置，反过来说不一定对）   
- 密度可达：对于 xi 和 xj, 若存在样本序列 p1,p2,...,pn,其中，p1=xi, pn=xj, 且 p(i+1) 由 pi 密度直达，则称xj由xi密度可达。(注：序列中的 p1,p2,...,p(n-1) 均为核心对象，因为要满足密度直达必须要是核心对象，对 pn 而言，其还不用是核心对象，因为他是最后一个。密度直达满足直递性，但不满足对称性)   
- 密度相连：对于 xi 和 xj,如果存在核心对象 xk，使 xi 和 xj 均由 xk 密度可达，则称 xi 和 xj 密度相连。(密度相连关系是满足对称性)   
- 簇：由密度可达关系导出的最大的密度相连样本集合。   
> 
- 举例加深理解：  
![](https://ftp.bmp.ovh/imgs/2020/12/ebfceb56502ca7ac.png)    
上图中，我们先指定 ϵ(圆的半径), MinPts=3。  
每个圆圈显示的是核心对象的 ϵ-邻域范围，x1 是核心对象，x2 由 x1 密度直达(在圆圈内的都是核心对象密度直达的)，x3 由 x1 密度可达，x3 和 x4 是密度相连。   
可以认为，DBSCAN 算法将所有点分为三类：核心点、非核心点(边界点)、异常点。(边界点在邻域内但周围样本点数量不达标，异常点不在领域内)  
![](https://ftp.bmp.ovh/imgs/2020/12/3abce8f81a9d68bd.png)  
>
#### (1.2) 实现过程：  
怎么聚类呢？  
(1) 指定ϵ, MinPts 值，遍历所有数据点，判断是否为核心对象,然后这些核心对象组成一个核心集合。  
(2) 从核心集合任意选择一个没有类别的核心对象作为初始点，然后找到所有这个核心对象能够密度可达的样本集合，形成一个聚类簇。  
(3) 排除掉已经在簇里面的核心对象，接着重复上述步骤，继续任意选择另一个没有类别的核心对象，去发展其密度可达的所有样本点，形成新的聚类簇。(重复上一步)  
(4) 直到所有核心对象被归类到对应的簇，停止发展下线。在簇里面的就是分类好的数据点，其余的就是异常值点。  


### 二. 算法优缺点：
#### (2.1) 优点：  
- 不需要事先给定一个K值（KMmeans 需要事先确定一个K值）    
- 适用于任意形状的稠密数据集，不单单是凸数据集(KMmeans 仅对凸数据集聚类效果好)    
- 能够有效的发现噪声点、异常值，可用于异常值检测
- 初始值的选择对聚类结果无影响，聚类结果几乎不依赖于结点遍历顺序，没有偏倚（初始质心的选择影响 KMmeans 迭代速度及效果）
>
#### (2.2) 缺点：  
- 当聚类的密度不同或或类间的距离相差很大时，DBSCAN 的性能会不如其他算法。在类中的数据分布密度不均匀时，假定 MinPts 不变，当 ϵ 较小时，密度小的点群会被划分成多个簇；当 ϵ 较大时，会使得距离较近且密度较大的点群被合并成一个簇。   
- 算法涉及两个参数：距离阈值 ϵ，邻域样本数阈值 MinPts。需要联合调参，不同的参数组合对最后的聚类效果有较大影响
- 面对高维数据容易溢出，算起来慢（可以先做降维）
- 该算法的运行速度要比 KMeans 算法慢一些
- 某些样本点可能到两个核心对象的距离都小于 ϵ ，但是这两个核心对象由于不是密度直达，且不属于同一个聚类簇，DBSCAN算法会采用先来后到的方法来判定这些样本点属于哪个簇，所以这个算法存在一定的不稳定性  
- 对于不是明显分离的簇，DBSCAN 可能会合并这些有重叠的簇，而 KMeans 倾向于区分开这些簇  
>

### 三. 举栗子：
下面用sklearn的make_moons方法来生成 DBSCAN 聚类算法的测试数据：  
```
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Hush
# @Software: PyCharm

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

X1, Y1 = datasets.make_moons(n_samples=5000, noise=0.07, random_state=678)
plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1)
plt.title('模拟数据')
plt.show()
```
![](https://ftp.bmp.ovh/imgs/2020/12/ff4bd24fda9de5f6.png)  
>
#### (3.1) 用 KMeans 聚类试试水
```
# 先用 KMeans 算法来分类
y1_model = KMeans(n_clusters=2, random_state=666)
y1_pred = y1_model.fit_predict(X1)
plt.scatter(X1[:, 0], X1[:, 1], c=y1_pred)
plt.show()
```
![](https://ftp.bmp.ovh/imgs/2020/12/91147fdefe45fdb6.png)  
很明显，KMeans不能很好地聚类，下面看看DBSCAN的情况。

#### (3.2) DBSCAN 聚类
```
# 现用 DBSCAN 算法来分类
y2_model = DBSCAN(eps=0.5, min_samples=5, metric='euclidean')  # 都是默认参数，半径设0.5，Minpts=5,采用欧氏距离
y2_pred = y2_model.fit_predict(X1)
plt.scatter(X1[:, 0], X1[:, 1], c=y2_pred)
plt.show()
```
![](https://ftp.bmp.ovh/imgs/2020/12/1947404f449dc50b.png)  
修改一下参数，可以改eps，也可以改min_samples。这里改eps.  
```
#  DBSCAN 修改参数eps
y2_model = DBSCAN(eps=0.1, min_samples=5, metric='euclidean')  # 都是默认参数，半径设0.5，Minpts=5,采用欧氏距离
y2_pred = y2_model.fit_predict(X1)
plt.scatter(X1[:, 0], X1[:, 1], c=y2_pred)
plt.show()
```
![](https://ftp.bmp.ovh/imgs/2020/12/eb988f99937dbc39.png)  
>
参考资料：  
西瓜书
[B站视频](https://www.bilibili.com/video/BV1j4411H7xv?p=1)  
大佬Blog,[(1)](https://www.cnblogs.com/pinard/p/6208966.html)、知乎多个回答
聚类效果可视化体验(某位外国大佬写的体验网站)[地址](https://www.naftaliharris.com/blog/visualizing-dbscan-clustering/)  
![](https://ftp.bmp.ovh/imgs/2020/12/1187054a9d252826.png)  
