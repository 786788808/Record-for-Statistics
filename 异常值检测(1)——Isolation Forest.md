## 异常检测 Anomaly Detection

### 背景：    
在建模前，需要做异常值检测(离群点检测)。以前建模的时候，一般用正态分布的三个标准差法则、箱线图来看是否存在异常值。但是对于复杂问题，总感觉有点不稳，于是看了下异常值检测方面的东西，记录一下。     

### 一. 异常值检测原因：  
- 特征工程过程中要剔除异常值，不然要影响建模效果(比如，KMeans算法)    
- 对没有标记输出的特征数据做筛选，找出异常值  
- 监督学习中，做二分类时，如果类别失衡，可以考虑用非监督的异常值检测算法来做(比如，检测患者是否患肺癌、判断用户是否存在欺诈行为)  

常见应用场景：网络安全中的攻击检测、疾病检测、金融交易欺诈检测、噪声数据过滤等等。    

有很多方法可以做异常值检测：KNN, PCA, SVM, Isolation Forest, LOF, DBSCAN等等，下面讲几种：  

### 二. 方法一：Isolation Forest 算法
是一种适用于**连续型数据**的**无监督**异常检测方法，主要是利用集成学习的思路来做异常点检测，具有线性时间复杂度，且精准度较高，在处理大数据时速度快，在业界应用广泛。       
"异常值"的两个假设：
> (1) 异常数据跟样本中大多数样本不太一样(疏离程度)(different)    
> (2) 异常数据在整体数据样本中占比少(few)    
>
异常值定义：   
> 容易被孤立的点 (more likely to be separated)，满足以下条件：a.分布稀疏 b.离密度高的群体较远。       
> 在特征空间里，分布稀疏的区域表示事件发生在该区域的概率很低，因而可以认为落在这些区域里的数据是异常的。    

#### (2.1) 算法思想：  
给定训练数据 X={x1, x2, …, xn}, 有 n 个样本点，数据维度为 d ，下面会用到多个 Isolation Tree 不断划分样本点。  
首先，算法随机地抽取某一特征A，并在该特征中随机选择一个值a，将样本划分为左右两个分支(大于等于a的在左支，小于a的在右支)  
然后，不断在每个分支重复上述步骤  
直到(满足以下条件之一)：  
- 每个叶子节点都只包含一个数据点(或者有多个一样的样本点)  
- 二叉树达到最高深度
则停止递归，结束算法。  

数据点在二叉树中所处的深度反映数据的‘疏离’程度。在这种随机分割的策略下，异常点通常具有较短的路径。  

个人理解就是，对于正常点需要多次分割才会被成功孤立(因为所处的簇密度高)，对于异常值只需要较少的分割次数就可以被成功孤立(因为所处的簇密度低)。  
以样本点的深度去判断即可。  

#### (2.2) 算法缺点：  
面对高维数据，处理能力欠缺，需要先做降维或者用其他方法，比如 one class SVM
如果训练样本中异常值较多，会违背算法的基本假设，导致检测效果不好

#### (2.3) 举栗子：  
[sklearn用法:](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html#sklearn.ensemble.IsolationForest)  
参数:  
Isolation Forest 算法主要有两个参数：一个是二叉树的数量；另一个是训练单棵 iTree 时候抽取样本的数目。实验表明，当设定为 100
棵树，抽样样本数为 256 条时候，IF 在大多数情况下就已经可以取得不错的效果。这也体现了算法的简单、高效。  
- n_estimators: int, 默认值100。基学习器的数量，即算法中，树的数量。
- max_samples： 'auto', int or float, default='auto'。最大样本数量，即训练每个基学习器的样本的数量。(论文提到采样大小超过256效果就提升不大了，并且越大还会造成计算时间上的浪费)  
> If int, then draw max_samples samples.    
> If float, then draw max_samples * X.shape[0] samples.    
> If 'auto', then max_samples=min(256, n_samples).  
>> If max_samples is larger than the number of samples provided, all samples will be used for all trees (no sampling).  
- contamination: 'auto' or  float, default=’auto’。数据污染问题，表示数据集中异常值的期望比例或者说是异常值的比例阈值。  
- random_state: 可设置复现
>
Methods:  
fit_predict(X[, y])： 训练集放进去可输出是否异常值的结果
predict(X)：判断是否异常值，+1 表示正常样本，-1表示异常样本。
decision_function(X)： 返回样本的异常评分。 值越小表示越有可能是异常样本。

简单例子：  
```
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Hush

import pandas as pd
from sklearn.ensemble import IsolationForest

X1 = [[-1.1], [0.3], [0.5], [99], [0.2], [100], [0.7]]
iso_model = IsolationForest(n_estimators=100, max_samples='auto',
                            contamination='auto', random_state=777)
iso_model.fit(X1)
y_pred_train = iso_model.predict(X1)
df = pd.DataFrame(X1, columns=['data'])
df['scores'] = iso_model.decision_function(X1)
df['is_outliers'] = iso_model.predict(X1)
print('——————1表示正常值，-1表示异常值——————')
print(df)

outliers = df.loc[df['is_outliers']==-1]
print('————————————异常值：————————————\n', outliers)
```
![](https://ftp.bmp.ovh/imgs/2020/12/9bde18830169f83a.png)
![](https://ftp.bmp.ovh/imgs/2020/12/3d161c72e0325218.png)  
>
### 三. 方法二：one calss SVM 算法
暂时不用，后面补。  

### 四. 方法三：DBSCAN 算法
直接去看DBSCAN那篇，DBSCAN的主要目的是做聚类，而异常值检测有点附赠礼品的意思。DBSCAN 算法基于密度将数据点聚类，然后将所有的数据点划分为：核心对象、边界点、异常值点。  
在建模前，需要定义： 距离 ℇ、邻域内包含的最小点数 Minpts  
- 核心点：在距离 ℇ 内至少具有最小包含点数（Minpts）的数据点
- 边界点： 核心点的距离ℇ内邻近点，但包含的点数小于最小包含点数（Minpts）
- 异常值点： 非核心点、非边界点，两个条件都满足不了的点



