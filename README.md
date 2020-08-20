# basic_AI_algorithm
Include Supervised learning:Logistic Regression、Multilayer perceptron 、 Deep Convolutional Network and Long Short-term Memory.And Unsupervised learning:Auto Encoders、Denoising Autoencoders、Stacked Denoising Auto-Encoders、Restricted Boltzmann Machines、Deep Belief Networks.

## Dateset
使用Mnist数据集
> 数据集描述：

每张图是28 * 28的大小，在训练集中一共60000个样本，在测试集中一共是10000个样本，一共是10类手写数字。
其中在pytroch框架中torchvision.datasets中封装了Mnist数据集，可使用pytorch中的DataLoader来读取随机读取、显示和处理数据集。

为了更直观的理解数据集，利用matplotlib随机显示了16个训练集中的手写数字以及其对应标签，如下图：
<div align=center><img src="https://github.com/Cathy-t/basic_AI_algorithm/blob/master/dataset.png"></div>

## Supervised_learning

 [Logistic Regression](https://github.com/Cathy-t/basic_AI_algorithm/tree/master/Supervised_learnin/Logistic_Regression)  | 逻辑回归 | [README](https://github.com/Cathy-t/basic_AI_algorithm/blob/master/Supervised_learnin/Logistic_Regression/README.md)
 
 [MLP](https://github.com/Cathy-t/basic_AI_algorithm/tree/master/Supervised_learnin/Multilayer%20perceptron)  | 多层感知机 | [README]
 
 [LeNet](https://github.com/Cathy-t/basic_AI_algorithm/tree/master/Supervised_learnin/Deep_CNN)  | 经典卷积神经网络之一 | [README]
 
 [LSTM](https://github.com/Cathy-t/basic_AI_algorithm/tree/master/Supervised_learnin/lstm)  | 长短期记忆网络  | [README]
 
## Unsupervised_learning

 [Auto_Encoder](https://github.com/Cathy-t/basic_AI_algorithm/tree/master/Unsupervised_learnin/Auto_Encoder)  | 自动编码器  | [README]
 
 [Denoising_Autoencoder](https://github.com/Cathy-t/basic_AI_algorithm/tree/master/Unsupervised_learnin/Denoising_Autoencoder)  | 降噪自动编码器  | [README]
 
 [Stacked_Denoising_Auto-Encoders](https://github.com/Cathy-t/basic_AI_algorithm/tree/master/Unsupervised_learnin/Stacked_Denoising_Auto-Encoders)  | 堆叠降噪自动编码器  | [README]
 
 [Restricted_Boltzmann_Machines](https://github.com/Cathy-t/basic_AI_algorithm/tree/master/Unsupervised_learnin/Restricted_Boltzmann_Machines)  | 受限玻尔兹曼机  | [README]
 
 [Deep_Belief_Networks](https://github.com/Cathy-t/basic_AI_algorithm/tree/master/Unsupervised_learnin/Deep_Belief_Networks)  | 深度置信网络  | [README]
