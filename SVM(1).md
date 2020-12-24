最近在看SVM，看了几遍，还是晕乎乎，先大概记录用的时候怎么用，后面再写原理吧  
![](https://gimg2.baidu.com/image_search/src=http%3A%2F%2Fimg.bqatj.com%2Fimg%2Feda21be90612d87e.jpg&refer=http%3A%2F%2Fimg.bqatj.com&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=jpeg?sec=1611386373&t=d8b9939603ff5befe69cee7fbf148c35)    
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
- 对于线性问题，采用线性核搞定，因为如果这时候用高斯核，需要浪费时间来调参。对于非线性问题，直接采用高斯核(RBF)来解决，一般都会效果比较好(当然前提是要好好调参)  


