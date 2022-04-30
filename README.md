# TextCNN

>target: 在TensorFlow中实现CNN进行文本分类

>参考网站：https://github.com/dennybritz/cnn-text-classification-tf

>电影评论数据集下载：http://www.cs.cornell.edu/people/pabo/movie-review-data/

>了解用于NLP的卷积神经网络 http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/
 
>英文和中文的区别就是分词的过程，中文一般使用jieba,或者word2vec(gensim库） 

## Training:
>python train.py --help 查看参数的设置

>python train.py  采用默认参数开始训练

## Evaluation:
>python evaluation.py --help 查看参数的设置

>python evaluation.py --checkpoint_dir runs/1651319732/checkpoints/ 
指定模型位置，开始测试

# requirements
>tensorflow=1.0.0(high is okay.)

>python3