## 线性判别分析 LDA

目录：  
- LDA 概念
- LDA 与 PCA 同异
- sklearn 用法

### 一. LDA 概念
线性判别分析(Linear Discriminant Analysis)，跟 PCA 一样，也是一种很常见的降维算法。同时它还可以用作分类算法。
但是 LDA 是一种有监督的降维技术，它需要样本具有类别信息。
LDA 的目标可以理解为：投影后类内方差最小，类间方差最大。即 LDA 要找能使类内方差小，类间方差大的方向去投影。  
投影后：  
(1) 同一类别数据的投影点尽可能地接近  
(2) 不同类别的数据的类别中心之间的距离尽可能大

### 二. LDA 与 PCA 同异
#### 2.1 同：　　　　
- 都是降维算法　  　　　
- 都要用到矩阵特征分解　  　　　
- 都要假设数据分布符合高斯分布　  　　
　
#### 2.2 异  
- LDA 属于有监督算法，PCA属于无监督算法　　　
- LDA 选择投影后“类内方差小，类间方差大”的方向，使得投影后的维度具有判别性，不同类别的数据尽可能的分开；PCA 选择投影后具有最大方差的方向，因为方差越大，则包含的信息越多
- LDA 对降维后的维度数有限制，最多降到维度数：类别数(k-1)，而 PCA 没限制
- LDA可降维还可分类


### 三. sklearn 用法
用鸢尾花数据：  
```
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt

iris_df = datasets.load_iris()
X = iris_df.data
Y = iris_df.target
print('原始数据集大小：', X.shape)
lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit(X,Y)
X_lda = lda.transform(X)
plt.scatter(X_lda[:, 0], X_lda[:, 1],marker='o',c=Y)
plt.show()
```
输出：  
![](https://ae04.alicdn.com/kf/U1c325be9d8344ce19e949c889b51b97bU.jpg)

参考资料：  
https://www.zhihu.com/question/35666712  
https://zhuanlan.zhihu.com/p/271917978  
https://www.cnblogs.com/pinard/p/6244265.html  
