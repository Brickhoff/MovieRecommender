# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 17:24:39 2019

@author: Xuezhang Wu
"""

import tensorflow as tf
import os
import pickle
from sklearn.model_selection import train_test_split
import numpy as np

title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig = pickle.load(open('F:/movieRecommender/preprocess.p', mode='rb'))
print("data loaded")

def save_params(params):
    """
    Save parameters to file
    """
    with open('params.p', 'wb') as f:
        pickle.dump(params, f)


def load_params():
    """
    Load parameters from file
    """
    with open('params.p', mode='rb') as f:
        return pickle.load(f)

#嵌入矩阵的维度
embed_dim = 32
# 下面之所以要 +1 是因为编号和实际数量之间是差 1 的
#用户ID个数
uid_max = max(features.take(0,1)) + 1 # 6040
#性别个数
gender_max = max(features.take(2,1)) + 1 # 1 + 1 = 2
#年龄类别个数
age_max = max(features.take(3,1)) + 1 # 6 + 1 = 7
#职业个数
job_max = max(features.take(4,1)) + 1# 20 + 1 = 21

#电影ID个数
movie_id_max = max(features.take(1,1)) + 1 # 3952
#电影类型个数，有个<PAD>
movie_categories_max = max(genres2int.values()) + 1 # 18 + 1 = 19
#电影名单词个数
movie_title_max = len(title_set) # 5216

#对电影类型嵌入向量做加和操作的标志，考虑过使用mean做平均，但是没实现mean
combiner = "sum"

#电影名长度，做词嵌入要求输入的维度是固定的，这里设置为 15
# 长度不够用空白符填充，太长则进行截断
sentences_size = title_count # = 15
#文本卷积滑动窗口，分别滑动2, 3, 4, 5个单词
window_sizes = {2, 3, 4, 5}
#文本卷积核数量
filter_num = 8

#电影ID转下标的字典，数据集中电影ID跟下标不一致，比如第5行的数据电影ID不一定是5
movieid2idx = {val[0]:i for i, val in enumerate(movies.values)}

# Hyperparameters
# Number of Epochs
num_epochs = 5
# Batch Size
batch_size = 256
dropout_keep = 0.5
# Learning Rate
learning_rate = 0.0001
# Show stats for every n number of batches
show_every_n_batches = 50

save_dir = 'F:/movieRecommender/save'

# inputs
def get_inputs():
    uid = tf.placeholder(tf.int32, [None, 1], name="uid")
    user_gender = tf.placeholder(tf.int32, [None, 1], name="user_gender")
    user_age = tf.placeholder(tf.int32, [None, 1], name="user_age")
    user_job = tf.placeholder(tf.int32, [None, 1], name="user_job")
    
    movie_id = tf.placeholder(tf.int32, [None, 1], name="movie_id")
    # 电影种类中要去除<PAD>，所以-1
    movie_categories = tf.placeholder(tf.int32, [None, movie_categories_max-1], name="movie_categories")
    movie_titles = tf.placeholder(tf.int32, [None, 15], name="movie_titles")
    targets = tf.placeholder(tf.int32, [None, 1], name="targets")
    LearningRate = tf.placeholder(tf.float32, name = "LearningRate")
    dropout_keep_prob = tf.placeholder(tf.float32, name = "dropout_keep_prob")
    return uid, user_gender, user_age, user_job, \
            movie_id, movie_categories, movie_titles, targets, \
                LearningRate, dropout_keep_prob

# Construct the neural network

# define user embedding matrix
# 其实用户的特征总数量是 128
# 但是作者在下面的代码中并没有这样做，而是由的特征设置为了 16，这样可能是因为作者的电脑性能比较差
# 我们先尝试这样的设置，如果计算资源允许，我们在后面再次测试的时候全部设置为 32
def get_user_embedding(uid, user_gender, user_age, user_job):
    with tf.name_scope("user_embedding"):
        # 下面的操作和情感分析项目中的单词转换为词向量的操作本质上是一样的
        # 用户的特征维度设置为 32
        # 先初始化一个非常大的用户矩阵
        # tf.random_uniform 的第二个参数是初始化的最小值，这里是-1，第三个参数是初始化的最大值，这里是1
        uid_embed_matrix = tf.Variable(tf.random_uniform([uid_max, embed_dim], -1, 1), 
                                       name = "uid_embed_matrix")
        # 根据指定用户ID找到他对应的嵌入层
        uid_embed_layer = tf.nn.embedding_lookup(uid_embed_matrix, uid, 
                                                 name = "uid_embed_layer")
    
        # 性别的特征维度设置为 16
#         gender_embed_matrix = tf.Variable(tf.random_uniform([gender_max, embed_dim // 2], -1, 1), 
#                                           name= "gender_embed_matrix")
        gender_embed_matrix = tf.Variable(tf.random_uniform([gender_max, embed_dim], -1, 1), 
                                          name= "gender_embed_matrix")
        gender_embed_layer = tf.nn.embedding_lookup(gender_embed_matrix, user_gender, 
                                                    name = "gender_embed_layer")
        
        # 年龄的特征维度设置为 16
#         age_embed_matrix = tf.Variable(tf.random_uniform([age_max, embed_dim // 2], -1, 1),
#                                        name="age_embed_matrix")
        age_embed_matrix = tf.Variable(tf.random_uniform([age_max, embed_dim], -1, 1),
                                       name="age_embed_matrix")
        age_embed_layer = tf.nn.embedding_lookup(age_embed_matrix, user_age, 
                                                 name="age_embed_layer")
        
        # 职业的特征维度设置为 16
#         job_embed_matrix = tf.Variable(tf.random_uniform([job_max, embed_dim // 2], -1, 1), 
#                                        name = "job_embed_matrix")
        job_embed_matrix = tf.Variable(tf.random_uniform([job_max, embed_dim], -1, 1), 
                                       name = "job_embed_matrix")
        job_embed_layer = tf.nn.embedding_lookup(job_embed_matrix, user_job, 
                                                 name = "job_embed_layer")
    # 返回产生的用户数据数据
    return uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer

# generate user fearture
def get_user_feature_layer(uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer):
    with tf.name_scope("user_fc"):
        #第一层全连接
        # tf.layers.dense 的第一个参数是输入，第二个参数是层的单元的数量
        uid_fc_layer = tf.layers.dense(uid_embed_layer, embed_dim, name = "uid_fc_layer", activation=tf.nn.relu)
        gender_fc_layer = tf.layers.dense(gender_embed_layer, embed_dim, name = "gender_fc_layer", activation=tf.nn.relu)
        age_fc_layer = tf.layers.dense(age_embed_layer, embed_dim, name ="age_fc_layer", activation=tf.nn.relu)
        job_fc_layer = tf.layers.dense(job_embed_layer, embed_dim, name = "job_fc_layer", activation=tf.nn.relu)
        
        #第二层全连接
        # 将上面的每个分段组成一个完整的全连接层
        user_combine_layer = tf.concat([uid_fc_layer, gender_fc_layer, age_fc_layer, job_fc_layer], 2)  #(?, 1, 128)
        # 验证上面产生的 tensorflow 是否是 128 维度的
        print(user_combine_layer.shape)
        # tf.contrib.layers.fully_connected 的第一个参数是输入，第二个参数是输出
        # 这里的输入是user_combine_layer，输出是200，是指每个用户有200个特征
        # 相当于是一个200个分类的问题，每个分类的可能性都会输出，在这里指的就是每个特征的可能性
        user_combine_layer = tf.contrib.layers.fully_connected(user_combine_layer, 200, tf.tanh)  #(?, 1, 200)
    
        user_combine_layer_flat = tf.reshape(user_combine_layer, [-1, 200])
    return user_combine_layer, user_combine_layer_flat

# define movieId embedding matrix
def get_movie_id_embed_layer(movie_id):
    with tf.name_scope("movie_embedding"):
        movie_id_embed_matrix = tf.Variable(tf.random_uniform([movie_id_max, embed_dim], -1, 1), name = "movie_id_embed_matrix")
        movie_id_embed_layer = tf.nn.embedding_lookup(movie_id_embed_matrix, movie_id, name = "movie_id_embed_layer")
    return movie_id_embed_layer

# sum
def get_movie_categories_layers(movie_categories):
    with tf.name_scope("movie_categories_layers"):
        movie_categories_embed_matrix = tf.Variable(tf.random_uniform([movie_categories_max, embed_dim], -1, 1), name = "movie_categories_embed_matrix")
        movie_categories_embed_layer = tf.nn.embedding_lookup(movie_categories_embed_matrix, movie_categories, name = "movie_categories_embed_layer")
        if combiner == "sum":
            movie_categories_embed_layer = tf.reduce_sum(movie_categories_embed_layer, axis=1, keep_dims=True)
    #     elif combiner == "mean":

    return movie_categories_embed_layer

# contruct TextCNN for Movie Title
def get_movie_cnn_layer(movie_titles):
    #从嵌入矩阵中得到电影名对应的各个单词的嵌入向量
    with tf.name_scope("movie_embedding"):
        movie_title_embed_matrix = tf.Variable(tf.random_uniform([movie_title_max, embed_dim], -1, 1), name = "movie_title_embed_matrix")
        movie_title_embed_layer = tf.nn.embedding_lookup(movie_title_embed_matrix, movie_titles, name = "movie_title_embed_layer")
        # 为 movie_title_embed_layer 增加一个维度
        # 在这里是添加到最后一个维度，最后一个维度是channel
        # 所以这里的channel数量是1个
        # 所以这里的处理方式和图片是一样的
        movie_title_embed_layer_expand = tf.expand_dims(movie_title_embed_layer, -1)
    
    #对文本嵌入层使用不同尺寸的卷积核做卷积和最大池化
    pool_layer_lst = []
    for window_size in window_sizes:
        with tf.name_scope("movie_txt_conv_maxpool_{}".format(window_size)):
            # [window_size, embed_dim, 1, filter_num] 表示输入的 channel 的个数是1，输出的 channel 的个数是 filter_num
            filter_weights = tf.Variable(tf.truncated_normal([window_size, embed_dim, 1, filter_num],stddev=0.1),name = "filter_weights")
            filter_bias = tf.Variable(tf.constant(0.1, shape=[filter_num]), name="filter_bias")
            
            # conv2d 是指用到的卷积核的大小是 [filter_height * filter_width * in_channels, output_channels]
            # 在这里卷积核会向两个维度的方向进行滑动
            # conv1d 是将卷积核向一个维度的方向进行滑动，这就是 conv1d 和 conv2d 的区别
            # strides 设置要求第一个和最后一个数字是1，四个数字的顺序要求默认是 NHWC，也就是 [batch, height, width, channels]
            # padding 设置为 VALID 其实就是不 PAD，设置为 SAME 就是让输入和输出的维度是一样的
            conv_layer = tf.nn.conv2d(movie_title_embed_layer_expand, filter_weights, [1,1,1,1], padding="VALID", name="conv_layer")
            # tf.nn.bias_add 将偏差 filter_bias 加到 conv_layer 上
            # tf.nn.relu 将激活函数设置为 relu
            relu_layer = tf.nn.relu(tf.nn.bias_add(conv_layer,filter_bias), name ="relu_layer")
            
            
            # tf.nn.max_pool 的第一个参数是输入
            # 第二个参数是 max_pool 窗口的大小，每个数值表示对每个维度的窗口设置
            # 第三个参数是 strides，和 conv2d 的设置是一样的
            # 这边的池化是将上面每个卷积核的卷积结果转换为一个元素
            # 由于这里的卷积核的数量是 8 个，所以下面生成的是一个具有 8 个元素的向量
            maxpool_layer = tf.nn.max_pool(relu_layer, [1,sentences_size - window_size + 1 ,1,1], [1,1,1,1], padding="VALID", name="maxpool_layer")
            pool_layer_lst.append(maxpool_layer)

    #Dropout层
    with tf.name_scope("pool_dropout"):
        # 这里最终的结果是这样的，
        # 假设卷积核的窗口是 2，卷积核的数量是 8
        # 那么通过上面的池化操作之后，生成的池化的结果是一个具有 8 个元素的向量
        # 每种窗口大小的卷积核经过池化后都会生成这样一个具有 8 个元素的向量
        # 所以最终生成的是一个 8 维的二维矩阵，它的另一个维度就是不同的窗口的数量
        # 在这里就是 2,3,4,5，那么最终就是一个 8*4 的矩阵，
        pool_layer = tf.concat(pool_layer_lst, 3, name ="pool_layer")
        max_num = len(window_sizes) * filter_num
        # 将这个 8*4 的二维矩阵平铺成一个具有 32 个元素的一维矩阵
        pool_layer_flat = tf.reshape(pool_layer , [-1, 1, max_num], name = "pool_layer_flat")
    
        dropout_layer = tf.nn.dropout(pool_layer_flat, dropout_keep_prob, name = "dropout_layer")
    return pool_layer_flat, dropout_layer

# fully connected layer
def get_movie_feature_layer(movie_id_embed_layer, movie_categories_embed_layer, dropout_layer):
    with tf.name_scope("movie_fc"):
        #第一层全连接
        movie_id_fc_layer = tf.layers.dense(movie_id_embed_layer, embed_dim, name = "movie_id_fc_layer", activation=tf.nn.relu)
        movie_categories_fc_layer = tf.layers.dense(movie_categories_embed_layer, embed_dim, name = "movie_categories_fc_layer", activation=tf.nn.relu)
    
        #第二层全连接
        movie_combine_layer = tf.concat([movie_id_fc_layer, movie_categories_fc_layer, dropout_layer], 2)  #(?, 1, 96)
        movie_combine_layer = tf.contrib.layers.fully_connected(movie_combine_layer, 200, tf.tanh)  #(?, 1, 200)
    
        movie_combine_layer_flat = tf.reshape(movie_combine_layer, [-1, 200])
    return movie_combine_layer, movie_combine_layer_flat



# 构建计算图 
    
# reset_default_graph 操作应该在 tensorflow 的其他所有操作之前进行，否则将会出现不可知的问题
# tensorflow 中的 graph 包含的是一系列的操作和使用到这些操作的 tensor
tf.reset_default_graph()
train_graph = tf.Graph()
# Graph 只是当前线程的属性，如果想要在其他的线程使用这个 Graph，那么就要像下面这样指定
# 下面的 with 语句将 train_graph 设置为当前线程的默认 graph
with train_graph.as_default():
    #获取输入占位符
    uid, user_gender, user_age, user_job, \
        movie_id, movie_categories, movie_titles, \
            targets, lr, dropout_keep_prob = get_inputs()
    #获取User的4个嵌入向量
    uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer = get_user_embedding(uid, user_gender, user_age, user_job)
    #得到用户特征
    user_combine_layer, user_combine_layer_flat = get_user_feature_layer(uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer)
    #获取电影ID的嵌入向量
    movie_id_embed_layer = get_movie_id_embed_layer(movie_id)
    #获取电影类型的嵌入向量
    movie_categories_embed_layer = get_movie_categories_layers(movie_categories)
    #获取电影名的特征向量
    pool_layer_flat, dropout_layer = get_movie_cnn_layer(movie_titles)
    #得到电影特征
    movie_combine_layer, movie_combine_layer_flat = get_movie_feature_layer(movie_id_embed_layer, 
                                                                                movie_categories_embed_layer, 
                                                                                dropout_layer)
    #计算出评分，要注意两个不同的方案，inference的名字（name值）是不一样的，后面做推荐时要根据name取得tensor
    # tensorflow 的 name_scope 指定了 tensor 范围，方便我们后面调用，通过指定 name_scope 来调用其中的 tensor
    with tf.name_scope("inference"):
        # 直接将用户特征矩阵和电影特征矩阵相乘得到得分，最后要做的就是对这个得分进行回归
        inference = tf.reduce_sum(user_combine_layer_flat * movie_combine_layer_flat, axis=1)
        inference = tf.expand_dims(inference, axis=1)

    with tf.name_scope("loss"):
        # MSE损失，将计算值回归到评分
        cost = tf.losses.mean_squared_error(targets, inference )
        # 将每个维度的 cost 相加，计算它们的平均值
        loss = tf.reduce_mean(cost)
    # 优化损失 
#     train_op = tf.train.AdamOptimizer(lr).minimize(loss)  #cost
# 在为 tensorflow 设置 name 参数的时候，是为了能在 graph 中看到什么变量进行了什么操作
    global_step = tf.Variable(0, name="global_step", trainable=False)
#     optimizer = tf.train.AdamOptimizer(lr)
    optimizer = tf.train.AdamOptimizer()
    gradients = optimizer.compute_gradients(loss)  #cost
    train_op = optimizer.apply_gradients(gradients, global_step=global_step)

# 自定义获取 batch 的方法
def get_batches(Xs, ys, batch_size):
    for start in range(0, len(Xs), batch_size):
        end = min(start + batch_size, len(Xs))
        yield Xs[start:end], ys[start:end]




import time
import datetime

# Train the network

losses = {'train':[], 'test':[]}

with tf.Session(graph=train_graph) as sess:
    
    #搜集数据给tensorBoard用
    # Keep track of gradient values and sparsity
    grad_summaries = []
    for g, v in gradients:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name.replace(':', '_')), g)
            # tf.nn.zero_fraction 用于计算矩阵中 0 所占的比重，也就是计算矩阵的稀疏程度
            sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name.replace(':', '_')), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
    grad_summaries_merged = tf.summary.merge(grad_summaries)
        
    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))
     
    # Summaries for loss and accuracy
    loss_summary = tf.summary.scalar("loss", loss)

    # Train Summaries
    train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    # Inference summaries
    inference_summary_op = tf.summary.merge([loss_summary])
    inference_summary_dir = os.path.join(out_dir, "summaries", "inference")
    inference_summary_writer = tf.summary.FileWriter(inference_summary_dir, sess.graph)

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    for epoch_i in range(num_epochs):
        
        #将数据集分成训练集和测试集，随机种子不固定
        train_X,test_X, train_y, test_y = train_test_split(features,  
                                                           targets_values,  
                                                           test_size = 0.2,  
                                                           random_state = 0)  
        
        train_batches = get_batches(train_X, train_y, batch_size)
        test_batches = get_batches(test_X, test_y, batch_size)
    
        #训练的迭代，保存训练损失
        for batch_i in range(len(train_X) // batch_size):
            x, y = next(train_batches)

            categories = np.zeros([batch_size, 18])
            for i in range(batch_size):
                categories[i] = x.take(6,1)[i]

            titles = np.zeros([batch_size, sentences_size])
            for i in range(batch_size):
                titles[i] = x.take(5,1)[i]

            feed = {
                uid: np.reshape(x.take(0,1), [batch_size, 1]),
                user_gender: np.reshape(x.take(2,1), [batch_size, 1]),
                user_age: np.reshape(x.take(3,1), [batch_size, 1]),
                user_job: np.reshape(x.take(4,1), [batch_size, 1]),
                movie_id: np.reshape(x.take(1,1), [batch_size, 1]),
                movie_categories: categories,  #x.take(6,1)
                movie_titles: titles,  #x.take(5,1)
                targets: np.reshape(y, [batch_size, 1]),
                dropout_keep_prob: dropout_keep, #dropout_keep
                lr: learning_rate}

            step, train_loss, summaries, _ = sess.run([global_step, loss, train_summary_op, train_op], feed)  #cost
            losses['train'].append(train_loss)
            train_summary_writer.add_summary(summaries, step)  #
            
            # Show every <show_every_n_batches> batches
            if batch_i % show_every_n_batches == 0:
                time_str = datetime.datetime.now().isoformat()
                print('{}: Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                    time_str,
                    epoch_i,
                    batch_i,
                    (len(train_X) // batch_size),
                    train_loss))
                
        #使用测试数据的迭代
        for batch_i  in range(len(test_X) // batch_size):
            x, y = next(test_batches)
            
            categories = np.zeros([batch_size, 18])
            for i in range(batch_size):
                categories[i] = x.take(6,1)[i]

            titles = np.zeros([batch_size, sentences_size])
            for i in range(batch_size):
                titles[i] = x.take(5,1)[i]

            feed = {
                uid: np.reshape(x.take(0,1), [batch_size, 1]),
                user_gender: np.reshape(x.take(2,1), [batch_size, 1]),
                user_age: np.reshape(x.take(3,1), [batch_size, 1]),
                user_job: np.reshape(x.take(4,1), [batch_size, 1]),
                movie_id: np.reshape(x.take(1,1), [batch_size, 1]),
                movie_categories: categories,  #x.take(6,1)
                movie_titles: titles,  #x.take(5,1)
                targets: np.reshape(y, [batch_size, 1]),
                dropout_keep_prob: 1,
                lr: learning_rate}
            
            step, test_loss, summaries = sess.run([global_step, loss, inference_summary_op], feed)  #cost

            #保存测试损失
            losses['test'].append(test_loss)
            inference_summary_writer.add_summary(summaries, step)  #

            time_str = datetime.datetime.now().isoformat()
            if batch_i % show_every_n_batches == 0:
                print('{}: Epoch {:>3} Batch {:>4}/{}   test_loss = {:.3f}'.format(
                    time_str,
                    epoch_i,
                    batch_i,
                    (len(test_X) // batch_size),
                    test_loss))

    # Save Model
    saver.save(sess, save_dir)  #, global_step=epoch_i
    print('Model Trained and Saved')
