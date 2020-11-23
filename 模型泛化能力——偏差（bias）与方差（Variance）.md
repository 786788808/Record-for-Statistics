建模后，我们总想知道这个模型好不好，泛化能力如何。而泛化误差就是用来衡量模型泛化能力的。  
![](https://pic4.zhimg.com/80/v2-286539c808d9a429e69fd59fe33a16dd_720w.jpg?source=1940ef5c)
E_d[(y_d - f_d(x))^2] &=E_d[(y_d -\bar{f(x)} + \bar{f(x)} - f_d(x))^2]\\ 
	&= E_d[(y_d -\bar{f(x)})^2] +  E_d[(\bar{f(x)}-f_d(x))^2] +0\\
	&= E_d[(y_d -\bar{f(x)})^2] +  E_d[(\bar{f(x)}-f_d(x))^2] \\
	&= E_d[(y_d -y+y -\bar{f(x)})^2] + E_d[(\bar{f(x)}-f_d(x))^2]\\
	&= E_d[(y_d -y)^2] + E_d[(y -\bar{f(x)})^2] + 0 + E_d[(\bar{f(x)}-f_d(x))^2]\\
	&= E_d[(y_d -y)^2] + E_d[(y -\bar{f(x)})^2]  + E_d[(\bar{f(x)}-f_d(x))^2]\\
	&= \epsilon^2 + bias^2+ var
