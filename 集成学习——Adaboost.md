## Adaboost
回归Boosting：Boosting方法训练基学习器时采用串行的方式，各个基学习器之间有依赖。  
![](https://ae03.alicdn.com/kf/U29b89572558c473eb237adacc6ec40f1y.jpg)    
它的基本思想是将基学习器层层叠加，每一层在训练的时候，对前一层基学习器分错的样本，给予更高的权重。测试时，根据各层分类器的结果的加权得到最终结果。这是一个不断迭代的过程，可以看作：Learn from your mistake.  
训练过程中，每个新的模型都会基于前一个模型的表现结果进行调整，这也解释了AdaBoost：自适应（adaptive）叫法的由来。  

### 一. Adaboost 在 Boosting 思想的基础上，在两方面做出了改变：  
- (1) 每一轮如何改变训练数据的权值或概率分布  
> 提高前一轮被弱学习器错误分类样本的权值，而降低被正确分类样本的权值(分类错误的样本在后续会得到更多的关注)   
- (2) 如何将弱学习器组合成一个强分类器  
> 对于结合策略，Adaboost采用加权多数表决法，即加大分类误差率小的弱分类器的权值，使其在表决中起较大的作用，减小分类误差率大的弱分类器的权值，使其在表决中起较小的作用。  
> 

### 二. 具体步骤： 
(分类情况):     
给定数据集 T={(x1,y1), (x2,y2) , … , (xn,yn)}, 最终分类器为 G(x)。  
1. 初始化样本权重 1/n，权值分布构成权值向量 D1 = {w11, w12 ,…, w1n}。开始时，每个样本的权重都应该是一样的；  
2. 在上述样本概率分布情况下，训练弱分类器 G1；    
3. 计算弱分类器 G(1) 的分类误差率 e1；  
4. 计算弱分类器 G(1) 的权重系数 α1(在结合策略里用到的权重)。二分类情况，<img src="https://latex.codecogs.com/gif.latex?\alpha_&space;{i}=\frac{1}{2}*log(\frac{1-e_{i}}{e_{i}})" title="\alpha_ {i}=\frac{1}{2}*log(\frac{1-e_{i}}{e_{i}})" />。该指标是弱分类器的重要度指标，当分类误差率 e1 越小，基本分类器在最终分类器中的作用越大；当分类误差率 e1 越大，基本分类器在最终分类器中的作用越小。       
5. 根据弱学习的学习误差率表现，更新训练样本集的权重分布 D2。被错误分类的样本权重变大，被正确分类的样本权重变小，使误分类样本在下一轮学习中起更大的作用。不改变所给的训练数据，而不断改变训练数据权值的分布，是训练数据在基本分类器的学习中起不同的作用，这是 Adaboost 算法的一大特点； 
6. 后续步骤，不断重复上述2-5步骤，直至迭代次数达到我们指定的某个值 T。  
7. 用加权表决法，形成最后的强学习器 G(x)。

### 三. Adaboost算法的另一种解释：  
模型是加法模型(组合策略)、损失函数为指数函数、学习算法为前向分布算法(逐步迭代)时的二分类学习算法。  


### 四.算法优缺点：
#### 4.1 优点：
- 作为分类器，分类器精度高
- 不容易过拟合

#### 4.2 缺点：
- 易受异常样本点影响。Adaboost会使得难于分类样本的权值呈指数增长，训练会偏向于这类样本点，造成偏倚。  
- 依赖于弱分类器，而弱分类器的训练时间较长


 ### 五. 弱分类器类型
 理论上，多种学习器都可以用。但一般都选择决策树或者神经网络来做弱学习器。  
 决策树一般是CART分类数或者回归树。  
