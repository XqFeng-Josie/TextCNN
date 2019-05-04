#itddsdttTextCNN


在TensorFlow中实现CNN进行文本分类

参考网站：https://github.com/dennybritz/cnn-text-classification-tf

完整笔记见http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/

数据集下载：http://www.cs.cornell.edu/people/pabo/movie-review-data/的电影评论数据

了解用于NLP的卷积神经网络 http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/
 
 

英文和中文的区别就是分词的过程，中文一般使用jieba,或者word2vec(gensim库） 


这里我们加入了三个滤波器区域大小：2,3和4，每个滤波器有2个滤波器。每个滤波器对句子矩阵执行卷积并生成（可变长度）特征映射。然后在每个地图上执行1-max池，即记录来自每个特征地图的最大数目。因此，从所有六个地图生成单变量特征向量，并且这六个特征被连接以形成倒数第二层的特征向量。最后的softmax层接收这个特征向量作为输入，并用它来分类句子; 这里我们假设二进制分类，因此描述了两种可能的输出状态。资料来源：Zhang，Y.，＆Wallace，B。（2015）。

