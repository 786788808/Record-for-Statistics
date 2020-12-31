### 背景:  
建模过程中，我们经常遇到高维数据，几十维、甚至几百、几千维，很有必要做特征选择，不然对计算机的计算能力求高，而且可能影响最后建模效果，考虑太多反而不好。        
那么，该怎么选择特征呢？看到部分人习惯用相关系数来做决定，但其实这只是其中一种方法。    
下面记录下三大类方法，平时建模的时候可以尝试多种方法，比较一下各种特征的情况。也因为特征选择，大家的模型各有特色。正所谓：“数据和特征决定了机器学习的上限，而模型和算法只是逼近这个上限而已”   

### 一. 常见的方法有(特征选择的形式)：
#### 1. 过滤法（Filter）：
按照发散性或者相关性对各个特征进行评分，设定阈值或者待选择阈值的个数，选择特征
#### 2. 包装法（Wrapper）：
根据目标函数，每次选择若干特征或者排除若干特征，直到选择出最佳的子集
#### 3. 嵌入法（Embedding) ：
先使用某些机器学习的算法和模型进行训练，得到各个特征的权值系数，根据系数从大到小选择特征。类似于 Filter 方法，但是是通过训练来确定特征的优劣
>
一般，我们从两个角度考虑特征选择问题，首先，从方差角度出发，**考虑特征的方差大小**。如果特征的方差很小，即这个特征的值基本集中于一个很小的范围内，那么这个特征对样本的区分没起啥作用。赶紧扔掉！！！     
另一方面，我们可以**考虑特征与目标值的相关性**，相关性高的保留，相关性低的剔除。上面的三大类都是从这一角度考虑的。   
>
### 三. 各方法介绍(特征选择的形式)：
#### 3.1 过滤法（Filter）
#### 3.1.1 方差法
计算所有特征的方差，然后设定一个阈值，只有特征的方差大于这个阈值时，才会被保留下来。   
返回值为特征选择后的数据    
```
from sklearn.feature_selection import VarianceThreshold

VarianceThreshold(threshold=4).fit_transform(X_train)  # threshold指定阈值大小；也可以先 fit() 再 transform()
```
>
#### 3.1.2 相关系数法
计算每个特征与目标值的相关系数(线性相关)，同上，设定一个阈值，只有相应的相关系数大于这个阈值，才会被保留下来。
注意，只有连续型变量才能计算这个值。  
返回值为特征选择后的数据    
用SelectKBest,指定参数k，表明返回最好的k个特征
```
from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr

#第一个参数为计算评估特征是否好的函数，该函数输入特征矩阵和目标向量，输出二元组（评分，P值）的数组，数组第i项为第i个特征的评分和P值。在此定义为计算相关系数
SelectKBest(lambda X, Y: array(map(lambda x:pearsonr(x, Y), X.T)).T, k=2).fit_transform(X_train, Y_train)
```
>
#### 3.1.3 卡方检验
卡方检验可以检验某个特征分布和目标值分布之间的相关性，然后给定卡方值阈值，只有卡方值大于这个阈值才会被保留  
选择 k 个最好的特征，返回值为特征选择后的数据    
```
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

#选择K个最好的特征，返回选择特征后的数据
SelectKBest(chi2, k=2).fit_transform(X_train, Y_train)
```
>
#### 3.1.4 互信息法
互信息值(信息增益)越大，说明特征和目标值之间的相关性越大，越需要保留
选择 k 个最好的特征，返回值为特征选择后的数据    
```
from sklearn.feature_selection import SelectKBest
from minepy import MINE
 
 #由于MINE的设计不是函数式的，定义mic方法将其为函数式的，返回一个二元组，二元组的第2项设置成固定的P值0.5
 def mic(x, y):
     m = MINE()
     m.compute_score(x, y)
     return (m.mic(), 0.5)

SelectKBest(lambda X, Y: array(map(lambda x:mic(x, Y), X.T)).T, k=2).fit_transform(X_train, Y_train)
```
>
#### 3.2 包装法（Wrapper）
#### 3.3 嵌入法（Embedding) 





