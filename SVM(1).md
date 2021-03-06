最近在看 SVM，看了几遍，还是晕乎乎，先大概记录用的时候怎么用，后面再写原理吧   
![](https://gimg2.baidu.com/image_search/src=http%3A%2F%2Fimg.bqatj.com%2Fimg%2Feda21be90612d87e.jpg&refer=http%3A%2F%2Fimg.bqatj.com&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=jpeg?sec=1611386373&t=d8b9939603ff5befe69cee7fbf148c35)    
### 一. 定义：
SVM 一般用于解决二类分类问题。它的基本思想是在特征空间中寻找间隔最大的分离超平面使数据得到高效的二分类。  
#### (1.1) 分类：
具体来讲，有三种情况（不加核函数的话就是个线性模型，加了之后才会升级为一个非线性模型）：
- 当训练样本线性可分时，通过硬间隔最大化，学习一个线性分类器，即线性可分支持向量机；
- 当训练数据近似线性可分时，引入松弛变量，通过软间隔最大化，学习一个线性分类器，即线性支持向量机；
- 当训练数据线性不可分时，通过使用**核技巧**及**软间隔**最大化，学习非线性支持向量机。
#### (1.2) 核函数分类：
![](https://ftp.bmp.ovh/imgs/2020/12/8e0048841532010d.png)

### 二. 优缺点：  
#### (1) 优点：  
- 有严格的理论支持，理论基础完善，可解释性强(所以这也是理解的难点，数分、高代、线代知识都用上了，使得它并没有 KMeans 算法这些这么容易理解；也不像神经网络黑盒操作，你也不知道怎么解释)
- 是一个凸优化问题，求得的解一定是全局最优，而不是局部最优
- 适用于线性、非线性问题
- 在高维样本空间的数据也能用 SVM (处理文本分类能力不错)，这是因为数据集的复杂度只取决于支持向量而不是数据集的维度，从而也避免了维数灾难这一问题 
- 对数据分布没有要求
  
#### (2) 缺点：
- 训练时间较长，采用 SMO 算法时(每次都需要挑选一对参数)，时间复杂度为 O(N^2) (其中 N 为训练样本的数量)
- 若用核技巧，如果需要存储核矩阵，则空间复杂度为 O(N^2)
- 模型预测时，预测时间与支持向量的个数成正比。当支持向量的数量较大时，预测计算复杂度较高
- 对多分类问题，是不断通过二分类来解决问题的
>
综上，SVM 目前只适合小批量样本的任务，无法适应百万甚至上亿样本的任务

### 三. sklearn.svm 用法：
#### (1) 分类：  
sklearn.svm 里包含 8 种 estimators，都是针对 SVM 问题展开：   
svm.LinearSVC、svm.LinearSVR、svm.NuSVC、svm.NuSVR、**svm.SVC**、**svm.SVR**、svm.OneClassSVM、svm.l1_min_c。    
- 以 SVC 结尾的，用于分类(classification)，以 SVR 结尾的是用于回(最后的两种不用管)    
- LinearSVC、LinearSVR 都仅支持线性核函数
- SVC、SVR 是我们常用来做 SVM 的
- 另外  NuSVC 分类 和 NuSVR 回归与单纯的 SVM、SVR 的区别在于对损失的度量方式不同。如果我们对训练集训练的错误率或者说支持向量的百分比有要求，可以选择 NuSVC 分类 和 NuSVR 回归(有参数来控制这个百分比)   

#### (2) SVC 分类：  
常用的是 SVC，下面讲 sklearn 里的 SVC 用法：  
[sklearn官方地址](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)  
语句：    
class sklearn.svm.SVC(*, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=- 1, decision_function_shape='ovr', break_ties=False, random_state=None)    

##### (2.1) 算法说明：      
- 该算法实现基于 libsvm，仅重写了算法了接口部分。  
- 拟合时间至少与样本数量成平方比例，对于大型数据集，请考虑使用 LinearSVC 或 SGDClassifier 代替。 
- 若数据集里有离散型特征，必须做数值化处理，因为 SVM 的自变量只能是连续性数值。     
- 对于线性问题，采用线性核搞定，因为如果这时候用高斯核，需要浪费时间来调参。对于非线性问题，直接采用高斯核(RBF)来解决，一般都会效果比较好(当然前提是要好好调参)。如果不知道是什么分布，直接用高斯核。    
##### (2.2) 参数说明：  
- **C**: 惩罚系数，默认值为 1，用的是 L2 正则化。一般用交叉验证来选择一个合适的值。
    - **C越大**，损失函数也越大，对错误例的惩罚程度越大，对错误例的容忍度比较大(就是你把更多的异常值考虑进来)，模型**容易过拟合**。所以，如果异常值点比较多，C 要适当减小一些。  
    - **C越小**，对错误例的容忍度比较小，模型**容易欠拟合**。  
- **kernel**: 设置内核，可选：'linear'(线性), 'poly'(多项式), 'rbf'(高斯核), 'sigmoid', 'precomputed', 默认值是 'rbf' 。也可以自定义核函数，看各位能力选择操作。
- **degree**: int，默认等于 3，多项式内核函数的阶数('poly')用，其余内核函数忽略。  
- **gamma**：如果是线性核，忽略这个参数。如果是多项式、高斯核、sigmoid 内核，那就要对这个参数调参了。有{'scale', 'auto'} or float 可选，如果设置为 auto，将=1/(特征维度数)
- **coef0**: 如果我们在 kernel 参数使用了多项式核函数 'poly'，或者 'sigmoid' 核函数，那么我们就需要对这个参数进行调参。默认等于 0。  
- **class_weight**: 指定各类别样本的的权重，因为数据集可能会样本失衡，使得模型偏向于占比多的类别。用这个参数指定各个样本的权重。如果样本均衡，使用默认的 'None' 即可；如果样本失衡，可以自己指定权重，或者用'balanced'，算法自动计算权重(样本量多的类别权重会低，样本量少的类别所权重会高，方便省事)。
- **decision_function_shape**: 
    - 对于二进制分类，将忽略该参数；
    - 对于多类别分类，可以选择'ovo'或者'ovr'，不同版本的 sklearn 对应的默认值不同，需要根据情况调整。  
        - OvR(one ve rest)：无论多少元分类，都看做二元分类。首先，对于第K类的分类决策，我们把所有第 K 类的样本作为正例，除了第 K 类样本以外的所有样本都作为负例，然后，在上面做二元分类，得到第 K 类的分类模型。其他类的分类模型获得以此类推。
        - OvO(one-vs-one)： 每次在所有的 T 类样本里面选择两类样本出来，记为 T1 类和 T2 类，然后把所有的输出为 T1 和 T2 的样本放在一起，把 T1 作为正例，T2 作为负例，进行二元分类，得到模型参数。我们一共需要T(T-1)/2次分类。    
        - 一般，多分类选择'ovo'，虽然速度比较慢，但大多数情况下分类效果较好。'ovr' 相对简单，但是大多数情况下分类效果没那么好。  
- **cache_size**: 缓存大小，默认值 200(200MB)。在大样本的时候，缓存大小会影响训练速度。如果机器内存大，用 500MB，甚至 1000MB，计算会快一点。  

### 四. 为什么要将求解 SVM 的原始问题转换为其对偶问题
- 使用对偶问题更易求解，当我们寻找带约束的最优化问题时，为了使问题变得易于处理，可以把目标函数和约束全部融入拉格朗日函数，再求解其对偶问题来寻找最优解
- 可以自然引入核函数，进而推广到非线性分类问题

### 五. 举栗子：
#### (1) 调参策略： 
下图为求解的原始形式和对偶化后的情况：    
![](https://ftp.bmp.ovh/imgs/2020/12/f7a50568df39425b.png)  
1. 对数据做归一化处理(包括训练集、测试集)(涉及或隐含距离计算的算法,KNN Means PCA也要)
2. 可以应用交叉验证和网格搜索来调参，GridSearchCV 结合两者来调最佳惩罚系数 C 和 gamma γ，方便省事（但是，在面对大数据集和多参数的情况下，非常耗时）      
3. 高斯核也可以模拟线性核，调参好的话效果可能更好。但是，面临维度过高，或者样本海量的情况下，选择线性核可能跟高斯核效果差不多，但是在速度和模型大小方面，线性核会有更好的表现。
>   
以高斯核 RBF 为例，因为经常遇到的情况是不知道数据是否线性可分，而且多项式核和 sigmod  需要调的超参数比较多，所以没头绪的时候用RBF
需要调整两个超参数：  
- **惩罚系数C**: 
> **当 C 比较大时**，会考虑更多的离群点，模型会有比较少的支持向量，模型会变得复杂，**容易过拟合**。   
- **核函数系数 gamma γ**:    
> 核函数K(x,z)=exp(−γ||x−z||^2)，(γ>0，需要自己调参)，γ主要定义了单个样本对整个分类超平面的影响。        
> **当γ比较大时**，整个模型的支持向量也会少，模型会变得更复杂，**容易过拟合**。     
>
综上:      
- 当 C 比较大， γ比较大时，模型的支持向量较少，模型比较复杂，容易过拟合    
- 当 C 比较小， γ比较小时，模型会变得简单，支持向量的个数会多，模型比较简单  

#### (2) 实例：   
数据集用经典的鸢尾花数据，没有缺失值，比较好处理：   
```
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Hush
# @Software: PyCharm

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

iris = load_iris()
# np.c_列拼接，np.r_行拼接
iris_df = pd.DataFrame(np.c_[iris.data, iris.target], columns=iris.feature_names+['flower_type'])
# print(iris_df.head(4))
print('数据集大小：', iris_df.shape)
print('缺失值数量\n', iris_df.isnull().sum())
print('各类别数量：\n', pd.pivot_table(iris_df, index='flower_type', values='sepal length (cm)', aggfunc=len))
```
拿到数据后，一般要先describe等等看数据分布情况，数据量、了解是否有缺失、异常值、类别是否平衡等等。  
[![rIhIgS.png](https://s3.ax1x.com/2020/12/27/rIhIgS.png)](https://imgchr.com/i/rIhIgS)
[![rI4SvF.png](https://s3.ax1x.com/2020/12/27/rI4SvF.png)](https://imgchr.com/i/rI4SvF)  
鸢尾花数据集比较小，没有缺失值，三个类别也均衡。  
```
# 先划分训练集、测试集
X_train, X_test, Y_train, Y_test = train_test_split(iris_df.iloc[:, :4], iris_df.iloc[:, -1],
                                                    train_size=0.75, random_state=666)
print('训练集大小：', X_train.shape)
print('训练集的各类别数量:\n', Y_train.value_counts())
```
[![rI42MF.png](https://s3.ax1x.com/2020/12/27/rI42MF.png)](https://imgchr.com/i/rI42MF)
```
scaled_X_train = StandardScaler().fit_transform(X_train)
param = {'kernel': ['rbf'], 'C':[0.01, 0.1, 1, 10, 100, 1000], 'gamma':[0.01, 0.1, 1, 10, 100, 1000]}
grid_search = GridSearchCV(estimator=SVC(class_weight='balanced', decision_function_shape='ovo'),
                           param_grid=param, cv=10)  # class_weight用'balanced'自动计算样本比例，decision_function_shape用'ovo'效果一般比较好，其余参数交给算法去测试得分
grid_search.fit(scaled_X_train, Y_train)
best_params = grid_search.best_params_  # 得出最佳参数组合
best_score = grid_search.best_score_  # 得出最佳得分
print('各参数组合得分\n', grid_search.cv_results_['mean_test_score'])  # 得出不同参数组合下10折的一个平均得分，直接用cv_results_可以看出每一折的得分
print('最佳参数及最高分数', best_params)
print('最高分数：%s' % format(best_score, '.3f'))
```
[![rIqRED.png](https://s3.ax1x.com/2020/12/28/rIqRED.png)](https://imgchr.com/i/rIqRED)
```
y_pred_1 = grid_search.predict(scaled_X_train)  # 预测训练集的结果
y_true_1 = Y_train
print('训练集的混淆矩阵\n', classification_report(y_true_1, y_pred_1, target_names=iris.target_names))  # 放到混淆矩阵看预测效果
```
[![rIqHDf.png](https://s3.ax1x.com/2020/12/28/rIqHDf.png)](https://imgchr.com/i/rIqHDf)
```
print('***********'*5)
print('看看模型在测试集的泛化能力：\n')
scaled_X_test = StandardScaler().fit_transform(X_test)
y_pred_2 = grid_search.predict(scaled_X_test)
y_true_2 = Y_test
print('测试集的混淆矩阵\n', classification_report(y_true_2, y_pred_2, target_names=iris.target_names))

```
[![rIqOUg.png](https://s3.ax1x.com/2020/12/28/rIqOUg.png)](https://imgchr.com/i/rIqOUg)   
从测试集效果来看，该模型还是不错的。     
150的数据量还是比较小的，可以在大数据集里看看效果。当然，SVM 对超大型数据的处理能力是明显不足的，计算复杂，耗时较久。如果是中小型数据还是可以考虑用 SVM 的，虽然集成学习已经很优秀了，但是这个算法的理论基础还是比较强的。    
对于该算法的理论掌握还不够，后期还需要好好学习一下。  



参考资料：    
[刘建平的支持向量机系列文章](https://www.cnblogs.com/pinard/p/6117515.html)
