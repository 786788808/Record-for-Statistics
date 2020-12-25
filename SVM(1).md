最近在看SVM，看了几遍，还是晕乎乎，先大概记录用的时候怎么用，后面再写原理吧   
![](https://gimg2.baidu.com/image_search/src=http%3A%2F%2Fimg.bqatj.com%2Fimg%2Feda21be90612d87e.jpg&refer=http%3A%2F%2Fimg.bqatj.com&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=jpeg?sec=1611386373&t=d8b9939603ff5befe69cee7fbf148c35)    
定义：
SVM 是一种二类分类模型。它的基本思想是在特征空间中寻找间隔最大的分离超平面使数据得到高效的二分类，具体来讲，有三种情况（不加核函数的话就是个线性模型，加了之后才会升级为一个非线性模型）：
- 当训练样本线性可分时，通过硬间隔最大化，学习一个线性分类器，即线性可分支持向量机；
- 当训练数据近似线性可分时，引入松弛变量，通过软间隔最大化，学习一个线性分类器，即线性支持向量机；
- 当训练数据线性不可分时，通过使用核技巧及软间隔最大化，学习非线性支持向量机。

优缺点：  
优点：  
由于SVM是一个凸优化问题，所以求得的解一定是全局最优而不是局部最优。不仅适用于线性线性问题还适用于非线性问题(用核技巧)。拥有高维样本空间的数据也能用SVM，这是因为数据集的复杂度只取决于支持向量而不是数据集的维度，这在某种意义上避免了“维数灾难”。理论基础比较完善(例如神经网络就更像一个黑盒子)。  
缺点：    
二次规划问题求解将涉及m阶矩阵的计算(m为样本的个数), 因此SVM不适用于超大数据集。(SMO算法可以缓解这个问题)只适用于二分类问题。(SVM的推广SVR也适用于回归问题；可以通过多个SVM的组合来解决多分类问题)  


sklearn.svm 里包含 8 种 estimators：   
svm.LinearSVC、svm.LinearSVR、svm.NuSVC、svm.NuSVR、svm.SVC、svm.SVR、svm.OneClassSVM、svm.l1_min_c。    
最后的两种不用管，前面有SVC结尾的，用于分类(classification)，SVR结尾的是用于回归的。    
LinearSVC、LinearSVR都仅支持线性核函数，SVC、SVR 是我们常用来做 SVM 的，另外  NuSVC分类 和 NuSVR回归与单纯的 SVM、SVR 的区别在于对损失的度量方式不同。如果我们对训练集训练的错误率或者说支持向量的百分比有要求，可以选择 NuSVC分类 和 NuSVR回归(有参数来控制这个百分比)。    

下面讲sklearn里的SVC用法：[sklearn官方地址](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)  
语句：  
class sklearn.svm.SVC(*, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=- 1, decision_function_shape='ovr', break_ties=False, random_state=None)    

>
说明：  
- 该算法实现基于libsvm，仅重写了算法了接口部分。  
- 拟合时间至少与样本数量成平方比例，对于大型数据集，请考虑使用LinearSVC或SGDClassifier代替。 
- 若数据集里有离散型特征，必须做数值化处理，因为SVM的自变量只能是连续性数值。     
- 对于线性问题，采用线性核搞定，因为如果这时候用高斯核，需要浪费时间来调参。对于非线性问题，直接采用高斯核(RBF)来解决，一般都会效果比较好(当然前提是要好好调参)。如果不知道是什么分布，直接用高斯核。    

参数：  
C: 惩罚系数，默认值为1。一般用交叉验证来选择一个合适的值。一般来说，如果噪音点较多时，C需要小一些。
C越大，损失函数也越大，对错误例的惩罚程度越大，模型容易过拟合。C越小，容易欠拟合。  
惩罚系数C即我们在之前原理篇里讲到的松弛变量的系数。它在优化函数里主要是平衡模型的复杂度和误分类率这两者之间的关系，可以理解为正则化系数。当C比较大时，我们的损失函数也会越大，这意味着我们不愿意放弃比较远的离群点。这样我们会有比较少的支持向量，也就是说支持向量和超平面的模型也会变得越复杂，也容易过拟合。反之，当C比较小时，意味我们不想理那些离群点，会选择较多的样本来做支持向量，最终的支持向量和超平面的模型也会简单。scikit-learn中默认值是1。
>
调参策略：  
1. 对数据做归一化处理(包括训练集、测试集)  
2. 应用交叉验证和网格搜索 GridSearch，在sklearn里可用GridSearchCV来调最佳惩罚系数 C 和 gamma γ  

















