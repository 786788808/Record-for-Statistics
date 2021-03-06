K近邻法(KNN)是一种很基本的机器学习算法，属于监督学习类算法，是一种简单易懂的方法，可用于回归和分类。比如我们要给点 A 做预测，做**分类**的时候，我们经常采用**少数服从多数**原则，A点最近的 K 个点属于哪个类最多，A点就属于那个类；做**回归**的时候，一般采用附近K个点的**平均值**作为A点的回归值。    
**注意：**  
很多人会把 KNN 与 KMeans 混合，记住 KNN：K 是指附近的 K 个点，有分类标签；KMeans：K 是指分成 K 类，无分类标签。  
![](https://gimg2.baidu.com/image_search/src=http%3A%2F%2Ffjmingfeng.com%2Fimg%2F0%2F0493698851%2F62%2Fdf825ad2a346cfb0be0a689aaed1f6ad%2F2374481702%2F6636934043.jpg&refer=http%3A%2F%2Ffjmingfeng.com&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=jpeg?sec=1612425698&t=0161a3c28d1e55bbc8318901e40c6501)  
## 目录：
- 原理
- 距离度量的实现方法
- 算法优缺点
- sklearn用法
- 应用举例
>

### 一. 原理
#### (1.1) 基本步骤：
简单来说，就是当预测一个数据点 A 的时候，根据它距离最近的 K 个点是什么类别来判断 x 属于哪个类别。    

主要实现过程：  
##### (1.1.1) 计算训练样本和测试样本中每个样本点的距离(常见的距离度量有欧式距离，马氏距离等)
##### (1.1.2) 对上面所有的距离值进行排序
##### (1.1.3) 选前k个最小距离的样本
##### (1.1.4) 根据这k个样本的标签进行投票，得到最后的分类类别

举个例子：  
我们手上有两种品种的芒果，从两个维度x y去衡量辨别品种，现有一个新的芒果A，想根据 KNN 去判定它属于哪一类别。三角形表示甲品种，爱心表示乙品种。  
过程：
- 取K值为3时，计算每个样本点到点A的距离，取最近的3个点作为评判点，现在有2个点是三角形，1个点是爱心，则判定点A属于甲品种。  
- 取K值为6时，计算每个样本点到点A的距离，取最近的6个点作为评判点，现在有4个点是爱心形，2个点是三角形，则判定点A属乙品种。
![](https://ftp.bmp.ovh/imgs/2021/01/0b5b9aa83df87f33.png)  

#### (1.2) 算法关键点：  
##### (1.2.1) K值的选择  
K值过大或过小，模型的泛化能力终将不好。    
当K值越小，模型越复杂，容易过拟合。    
当K值越大，模型越简单，容易欠拟合。     
可通过交叉验证选择合适的K值，这个是调参重点。        
##### (1.2.2) 距离的选择  
距离常见的有：曼哈顿距离、欧式距离、闵可夫斯基距离(一个通式，取2时，就是欧式距离)、切比雪夫距离、标准化欧式距离。    
一般直接选欧式距离即可。    
##### (1.2.3) 决策原则  
在分类算法里，一般采用少数服从多数原则   

#### (1.3) **注意：**
在计算距离的算法里，必须要做归一化or标准化。如果漏掉这一步，算法会偏向于取值范围较大的特征，影响模型效果。     

### 二. 距离度量的实现方法
#### (2.1) 蛮力实现  
我们上述的实现过程就是采用蛮力实现，逐个计算样本点到预测点的距离，取最近的K个点来决策。  
当样本量少、特征少的时候，用蛮力实现还是可以的。  
但是样本量达到几十万、特征几千个的时候，计算机可得算惨了，整个预测过程相对耗时(所以才叫蛮力实现)。  
于是有新的方法产生。  

#### (2.2) KD 树

#### (2.3) 球树

这部分暂时省略，为了提高效率，推出 KD 树、球树，如果数据量大，建议试试。  
详细可参考大神博客：https://www.cnblogs.com/pinard/p/6061661.html

### 三. 算法优缺点： 
KNN 算法中，我们不需要对数据分布做任何假设。其次，这是一个惰性算法，拿逻辑回归来说，逻辑回归需要训练再到预测，而 KNN 算法拿到数据后，基本没有训练一说，做预测的时候才开始算。  
KNN 相对其他算法，较易入门，相比SVM算法，这不要太简单了。  苍天啊！  
#### (3.1) 优点：
- 相对好理解，如果客户想要了解分类过程，用这个算法解释还是较好讲解的
- 可用于线性、非线性分类
- 训练时间复杂度比支持向量机之类的算法低，仅为O(n)
- 和朴素贝叶斯之类的算法比，对数据没有假设，准确度高，对异常点不敏感
- 对于重叠较多的待分样本，KNN单靠近邻做预测的方法比其他方法更合适，不用担心重叠影响
　　　　
#### (3.2) 缺点：
- 计算量大，特别是特征数非常多的时候
- 样本不平衡时，模型倾向于占比大的类别，不利于小比例类别的预测
- 采用KD树，球树之类的模型建立需要大量的内存
- 懒惰学习，基本上不学习，所以在做预测的时候，相比逻辑回归等算法会比较慢
- 与决策树模型相比，KNN算法可解释性较弱

### 四. sklearn用法：
sklearn.neighbors包中包含KNN的算法，可以直接用大神们造的轮子，掌握调参更为直接。  
下面是KNeighborsClassifier的介绍：    
[官网地址](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)   
语句：  
class sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, *, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None, **kwargs)

Parameters:  
- n_neighbors: int, default=5.  
> 考虑的近邻数量
- weights：{‘uniform’, ‘distance’} or callable, default=’uniform’.  
> 每个样本的近邻样本的权重。   
> 可以选择"uniform","distance" 或者自定义权重。  
>> 选择默认的"uniform"，意味着所有最近邻样本权重都一样，在做预测时一视同仁。
>> 选择"distance"，则权重和距离成反比例，即距离预测目标更近的近邻具有更高的权重，这样在预测类别或者做回归时，更近的近邻所占的影响因子会更加大。     
> 如果各类样本都在相对分开的簇中时，我们用默认的"uniform"；如果样本的分布比较乱，比较没规律，选择"distance"。如果用"distance"效果也不好，就自定义权重(可谓大神操作)。
- algorithm: {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, default=’auto’
> 计算近邻样本的算法。包含自动，球树，kd树，蛮力实现。 默认的auto会帮我们寻找最优算法，一般用这个也OK了。如果输入样本是稀疏的，算法最终都会用‘brute’。    
> 如果数据量很大、特征很多，计算会慢很多，可调为‘kd_tree’，加快计算速度。如果速度还是不够快，或者样本分布不是很均匀，可调成‘ball_tree’试试。      
- leaf_size： int, default=30.
> 控制停止建子树的叶子节点数量的阈值。使用brute蛮力实现的时候不会用到。在使用KD树或者球树时，会影响树的大小。  
> 当leaf_size越大，树会相对小一点，层数较少，建树时间较短。而leaf_size越小，树会大一点，层数较多，建树时间较长。  
> 叶大小传递给BallTree或KDTree。这会影响构造和查询的速度，以及存储树所需的内存。最佳值取决于问题的性质。   
> 如果样本很大，要适当增加这个值。不然这个值会相对设置过小，树会很大，从而造成过拟合。
- p：int, default=2.  
> Minkowski指标的功率参数。  
> 当p = 1时，代表用曼哈顿距离。  
> 当p = 2，代表使用欧式距离。默认用欧式距离，一般用默认就行。  
- metricst：default=’minkowski’.
> 距离度量方式，默认是’minkowski’，配合上面的P=2，就是欧式距离。  
- metric_params：dict, default=None    
> 主要是用于带权重闵可夫斯基距离的权重，一般不用管。  
- n_jobs：int, default=None  
> 指定多少个CPU进行运算，None表示1， 设置为-1表示使用所有处理器。主要影响运算速度，不影响算法本身。  

### 五. 应用举例：
拿鸢尾花数据来做个简单例子:  
```
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Hush
# @Software: PyCharm

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['FangSong']

iris = load_iris()
X = iris.data
Y = iris.target
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.75, random_state=666)
scale = StandardScaler()
x_train_scaled = scale.fit_transform(x_train)

k_list = range(1, 31)
k_error = []
for k in k_list:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_model, x_train_scaled, y_train, cv=6, scoring='accuracy')
    # print(scores)
    k_error.append(scores.mean())

# 从图挑出最佳K值
plt.plot(k_list, k_error)
plt.xlabel('K 值')
plt.ylabel('Accuracy')
plt.show()
```
![](https://ftp.bmp.ovh/imgs/2021/01/eafa27347de7f14a.png)  
由图看到，当 k = 11时，准确率最高。  
下面将K设置为11：  
```
best_knn = KNeighborsClassifier(n_neighbors=11)	
best_knn.fit(x_train_scaled, y_train)			
x_test_scaled = scale.fit_transform(x_test)
print(best_knn.score(x_test_scaled, y_test))	# 看看评分
```
![](https://ftp.bmp.ovh/imgs/2021/01/ce9ddc747d2d76cc.png)    

注意: KNN需要做好归一化处理，因为要计算距离，很多数据集里，如果漏掉这一步，一步错步步错  
