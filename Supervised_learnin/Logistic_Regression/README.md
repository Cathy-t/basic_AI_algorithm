# Logistic_Regression

该部分的实验分为两部分，一部分只使用nn.Linear()进行线性回归，另一部分将引入sigmoid作为每个神经元的激活函数进行逻辑回归。为了进行对比，这里统一设置num_epochs = 5，即迭代次数设置为5。实验结果采用训练中的loss变化曲线和对测试集中的数据进行测试后得到的混淆矩阵进行展示。前两行为训练过程中的loss变化曲线，从左到右，从上到下的学习率分别是0.001、0.005、0.01、0.1、0.5、1；后两行为混淆矩阵的可视化显示，从左到右，从上到下的学习率分别是0.001、0.005、0.01、0.1、0.5、1。

![](https://github.com/Cathy-t/basic_AI_algorithm/blob/master/Supervised_learnin/Logistic_Regression/lr.png)
![](https://github.com/Cathy-t/basic_AI_algorithm/blob/master/Supervised_learnin/Logistic_Regression/confusion_.png)

由于上述结果的实现并没有增加激活函数，于是接下来加入sigmoid函数来进行激活，结果如下所示。前两行为训练过程中的loss变化曲线，从左到右，从上到下的学习率分别是0.001、0.005、0.01、0.1、0.5、1；后两行为混淆矩阵的可视化显示，从左到右，从上到下的学习率分别是0.001、0.005、0.01、0.1、0.5、1。

![](https://github.com/Cathy-t/basic_AI_algorithm/blob/master/Supervised_learnin/Logistic_Regression/lr_a.png)
![](https://github.com/Cathy-t/basic_AI_algorithm/blob/master/Supervised_learnin/Logistic_Regression/confusino_a.png)

>Accuracy of the model on the 10000 test images: 

![](https://github.com/Cathy-t/basic_AI_algorithm/blob/master/Supervised_learnin/Logistic_Regression/learning_rate%E5%AF%B9LR%E7%9A%84%E5%BD%B1%E5%93%8D.png)
