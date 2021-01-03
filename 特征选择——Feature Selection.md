### 背景:  
建模过程中，我们经常遇到高维数据，几十维，甚至几百、几千维，很有必要做特征选择，不然计算机要算老半天，而且分分钟'维度灾难'，严重影响最后建模效果，所以乖乖剔除不必要的特征是重要的一步。            
那么，该怎么选择特征呢？看到部分人习惯用相关系数来做决定，但其实这只是其中一种方法。    
下面记录下三大类方法，平时建模的时候可以尝试多种方法，比较一下各种特征的情况。  
特征选择方法千千万，没有绝对的对与错。平时看到很多人会吐槽部分方法很简单，显得很low, 要上xgboost这类大佬才显得牛。但是，常言道：“数据和特征决定了机器学习的上限，而模型和算法只是逼近这个上限而已。”所以，还是好好修炼特征选择吧！！！     

### 一. 常见的方法有(特征选择的形式)：
一般，我们从两个角度考虑特征选择问题，首先，从方差角度出发，**考虑特征的方差大小**。如果特征的方差很小，即这个特征的值基本集中于一个很小的范围内，那么这个特征对样本的区分没起啥作用。赶紧扔掉！！！     
另一方面，我们可以**考虑特征与目标值的相关性**，相关性高的保留，相关性低的剔除。上面的三大类都是从这一角度考虑的。  
>
#### 1. 过滤法（Filter）：
按照发散性或者相关性对各个特征进行评分，设定阈值或者待选择阈值的个数，选择特征
#### 2. 包装法（Wrapper）：
根据目标函数，每次选择若干特征或者排除若干特征，直到选择出最佳的子集
#### 3. 嵌入法（Embedding) ：
先使用某些机器学习的算法和模型进行训练，得到各个特征的权值系数，根据系数从大到小选择特征。类似于 Filter 方法，但是是通过训练来确定特征的优劣
>
### 三. 各方法介绍(特征选择的形式)：
#### 3.1 过滤法（Filter）
##### 3.1.1 方差法
计算所有特征的方差，然后设定一个阈值，只有特征的方差大于这个阈值时，才会被保留下来。   
返回值为特征选择后的数据    
```
from sklearn.feature_selection import VarianceThreshold

VarianceThreshold(threshold=4).fit_transform(X_train)  # threshold指定阈值大小；也可以先 fit() 再 transform()
```
>
##### 3.1.2 相关系数法
计算每个特征与目标值的相关系数(线性相关)，同上，设定一个阈值，只有相应的相关系数大于这个阈值，才会被保留下来。
注意，只有连续型变量才能计算这个值。  
缺陷：如果特征与目标值存在很强的非线性关系，相关系数也检测不出来。  
返回值为特征选择后的数据    
用SelectKBest,指定参数k，表明返回最好的k个特征
```

from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr

#第一个参数为计算评估特征是否好的函数，该函数输入特征矩阵和目标向量，输出二元组（评分，P值）的数组，数组第i项为第i个特征的评分和P值。在此定义为计算相关系数
SelectKBest(lambda X, Y: array(map(lambda x:pearsonr(x, Y), X.T)).T, k=2).fit_transform(X_train, Y_train)
```
>
##### 3.1.3 卡方检验
卡方检验可以检验某个特征分布和目标值分布之间的相关性，然后给定卡方值阈值，只有卡方值大于这个阈值才会被保留  
选择 k 个最好的特征，返回值为特征选择后的数据    
```
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

SelectKBest(chi2, k=2).fit_transform(X_train, Y_train)
```
>
##### 3.1.4 互信息法
MIC 即：Maximal Information Coefficient 最大互信息系数
互信息值(信息增益)越大，说明特征和目标值之间的相关性越大，越需要保留
MIC可以用来衡量线性或非线性的相互关系
选择 k 个最好的特征，返回值为特征选择后的数据    
```
from sklearn.feature_selection import SelectKBest
from minepy import MINE
 
 def mic(x, y):
     m = MINE()
     m.compute_score(x, y)
     return (m.mic(), 0.5)

SelectKBest(lambda X, Y: array(map(lambda x:mic(x, Y), X.T)).T, k=2).fit_transform(X_train, Y_train)
```
>
#### 3.2 包装法（Wrapper）
Wrapper 会根据一个目标函数来逐步筛选特征，其中，最常用的方法是递归消除特征法 RFE (recursive feature elimination)。  
RFE 首先选择一个机器学习模型，将 n 维特征和目标值加入训练，一轮训练后，消除权值系数较低的特征。  
接着，在剩下的特征里，继续新一轮训练，不断剔除权值系数的特征。  
直到，剩下的特征数满足需求，停止训练。  
```
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# estimator指定基模型器，这里用LogisticRegression；n_features_to_select 指定保留特征个数
RFE(estimator=LogisticRegression(), n_features_to_select=3).fit_transform(X_train, Y_train) 
```

#### 3.3 嵌入法（Embedding) 
嵌入法也使用到机器学习模型，但是他是考虑所有特征，而不像包装法逐步剔除不重要的特征    
可以使用 L1、L2 正则化来选择特征，也可以选择决策树、GBDT 这类树模型来选择特征   
一般，可以得到**特征系数coef** 或者可以得到**特征重要度(feature importances)** 的算法才可做为嵌入法的基学习器
对于部分机器学习算法，其本身就具有对特征进行打分的机制，或者很容易将其运用到特征选择任务中，例如回归模型，SVM，决策树，随机森林等等

##### 3.3.1 基于惩罚项的特征选择法
下面选择逻辑回归和L1惩罚:  
```
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

SelectFromModel(LogisticRegression(penalty='l1', C=0.1)).fit_transform(X_train, Y_train)
```
还有SVM的svm.LinearSVC可以作为基模型，根据情况使用。  
对于SVM和逻辑回归，参数C控制稀疏性：C越小，被选中的特征越少。对于Lasso，参数alpha越大，被选中的特征越少。  
>

##### 3.3.2 基于树模型的特征选择法　　
树模型中GBDT也可用来作为基模型进行特征选择，使用feature_selection库的SelectFromModel类结合GBDT模型，来选择特征的代码如下：
```
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier

SelectFromModel(GradientBoostingClassifier()).fit_transform(iris.data, iris.target)
```

除了剔除特征，还可以构建新的特征，比如：特征相加、特征相减、相乘、相除，平方，立方等等，在一些比赛里，大神都会构建一些新的特征，所以在平时建模，也可以多去尝试新的模型，不仅仅基于现有的特征。  



