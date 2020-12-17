## 基于密度的聚类算法DBSCAN(Density-Based Spatial Clustering of Applications with Noise)
本文介绍基于密度的聚类算法，这类算法假定聚类结构能通过样本分布的紧密程度确定。  
通常情况下，密度聚类算法从样本密度的角度来考察样本之间的可连接性，并给予可连接样本不断扩展聚类簇以获得最终的聚类结果。  

>
### 一. DBSCAN基础概念及原理步骤：  
DBSCAN(具有噪声的基于密度的聚类方法)是其中一种基于密度的聚类算法,  它基于一组邻域参数参数(ϵ, MinPts)来刻画样本分布的紧密程度。   

![](https://ftp.bmp.ovh/imgs/2020/12/1187054a9d252826.png)  
看了西瓜书、[B站的一个视频(分三节)](https://www.bilibili.com/video/BV1j4411H7xv?p=1)以及一些blog，总结出来
这是一个循环迭代的算法，简单易懂。假定我们要对 N 个样本观测做聚类：  
(1) 假设聚为 K 类，并且随机选择 K 个点作为初始中心点；  
(2) 接下来，按照距离初始中心点最小的原则，把所有观测分到各中心点所在的类中；  
(3) 每类中有若干个观测，计算 K 个类中所有样本点的均值，作为第二次迭代的 K 个中心点；  
(4) 然后根据这个中心重复第 2、3 步，直到收敛（中心点不再改变或达到指定的迭代次数），聚类过程结束。  
下图展示最简单的聚类，分 2 类：  
![](https://ftp.bmp.ovh/imgs/2020/12/12cb6a37de432746.png)  
实际情况中，质心一般要迭代多次，才能达到相对最优情况。
可以以传销的方式去理解整个过程：
（1）首先，在整个数据集里随机地选取某个点A，然后根据我们设定的标准（半径以及圈内数量），评判A是否能开展一个新簇。假设半径为2、最小数量为4，如果点A达到以上标准，那么可以以它为基础开展一个新簇。
（2）假设点A圈住了5个点：B C D E F，现在这5个点就要去发展下线了。同样是以上述标准，去看他们能否发展下线。
（3）不断重复步骤（2），直至没有符合标准的下线了。这就构成了第一个簇C1。
（4）除了C1中的点，随机选取某个点，看它能否像点A一样去发展一个新簇。
>
### 二. 算法优缺点：
#### (2.1) 优点：  
- 不需要事先给定一个K值（KMmeans需要事先确定一个K值）    
- 适用于任意形状的稠密数据集，不单单是凸数据集（KMmeans仅对凸数据集聚类效果好）    
- 能够有效的发现噪声点、异常值，可用于异常值检测
- 初始值的选择对聚类结果无影响，聚类结果几乎不依赖于结点遍历顺序，没有偏倚（初始质心的选择影响KMmeans迭代速度及效果）
>
#### (2.2) 缺点：  
- 当聚类的密度不同或或类间的距离相差很大时，DBSCAN的性能会不如其他算法   
- 算法涉及两个参数：距离阈值ϵ，邻域样本数阈值MinPts。需要联合调参，不同的参数组合对最后的聚类效果有较大影响
- 面对高维数据容易溢出，算起来慢（可以先做降维）
- 该算法的运行速度要比 KMeans 算法慢一些
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
上述手肘法Elbow method需要人眼去观察最佳值，从而更智能的Gap statistic方法推出。只需要找出使Gap Statistic最大的K值即可。
安装包位置：https://www.cnpython.com/pypi/gapkmean
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
同样也是以簇内的稠密程度和簇间的离散程度来评估聚类的效果，选择使Calinski-Harabasz分数较大对应的k值
公式：![](https://ftp.bmp.ovh/imgs/2020/12/692a7a67f082412e.png)  
其中，m为训练集样本数，k为类别数。Bk为类别之间的协方差矩阵，Wk为类别内部数据的协方差矩阵。tr为矩阵的迹。
类别内部数据的协方差越小越好，类别之间的协方差越大越好，这样的Calinski-Harabasz分数会高。  
用法参考：sklearn.metrics.calinski_harabaz_score
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html
>
### 四. 举栗子：
下面用sklearn的make_blobs方法来生成聚类算法的测试数据：  

![](https://ftp.bmp.ovh/imgs/2020/12/169871e487d9afb4.png)
(4.1) 手肘法
![](https://ftp.bmp.ovh/imgs/2020/12/09adde7a83911ce0.png)


```
import pandas as pd



参考资料：  
西瓜书
[B站视频](https://www.bilibili.com/video/BV1j4411H7xv?p=1)  
大佬Blog,[(1)](https://www.cnblogs.com/pinard/p/6208966.html)、知乎多个回答
聚类效果可视化体验(某位外国大佬写的体验网站)[地址](https://www.naftaliharris.com/blog/visualizing-dbscan-clustering/)  
![](https://ftp.bmp.ovh/imgs/2020/12/1187054a9d252826.png)  
