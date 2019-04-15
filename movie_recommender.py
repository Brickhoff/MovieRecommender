# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 16:16:34 2019

@author: Xuezhang Wu
"""

import numpy as np
import tensorflow as tf

print(tf.__version__)

import pickle

import tkinter as tk


title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig = pickle.load(open('F:/movieRecommender/preprocess.p', mode='rb'))
print("data loaded")

#电影ID转下标的字典，数据集中电影ID跟下标不一致，比如第5行的数据电影ID不一定是5
movieid2idx = {val[0]:i for i, val in enumerate(movies.values)}
#电影名长度，做词嵌入要求输入的维度是固定的，这里设置为 15
# 长度不够用空白符填充，太长则进行截断
sentences_size = title_count # = 15

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

save_dir = 'F:/movieRecommender/save'
save_params((save_dir))
load_dir = load_params()

labels = []

movie_matrics = pickle.load(open('F:/movieRecommender/movie_matrics.p', mode='rb'))


def recommend_same_type_movie():
    
    movie_id_val = int(e.get())
    top_k = 20
    loaded_graph = tf.Graph()  #
    with tf.Session(graph=loaded_graph) as sess:  #
        # Load saved model
        loader = tf.train.import_meta_graph(load_dir + '.meta')
        #loader = tf.train.import_meta_graph(load_dir)
        loader.restore(sess, load_dir)
        
        norm_movie_matrics = tf.sqrt(tf.reduce_sum(tf.square(movie_matrics), 1, keep_dims=True))
        normalized_movie_matrics = movie_matrics / norm_movie_matrics

        #推荐同类型的电影
        probs_embeddings = (movie_matrics[movieid2idx[movie_id_val]]).reshape([1, 200])
        probs_similarity = tf.matmul(probs_embeddings, tf.transpose(normalized_movie_matrics))
        sim = (probs_similarity.eval())
    #     results = (-sim[0]).argsort()[0:top_k]
    #     print(results)
        
        #print("您看的电影是：{}".format(movies_orig[movieid2idx[movie_id_val]]))
        label1 = tk.Label(window, text='Chosen movie', bg='red')
        label1.pack()
        labels.append(label1)
        label2 = tk.Label(window, text= movies_orig[movieid2idx[movie_id_val]])
        label2.pack()
        labels.append(label2)
        label3 = tk.Label(window, text='Recommended Movie', bg='blue')
        label3.pack()
        labels.append(label3)
        #print("以下是给您的推荐：")
        p = np.squeeze(sim)
        p[np.argsort(p)[:-top_k]] = 0
        p = p / np.sum(p)
        results = set()
        while len(results) != 5:
            c = np.random.choice(3883, 1, p=p)[0]
            results.add(c)
        for val in (results):
            print(val)
            print(movies_orig[val])
            l = tk.Label(window, text= movies_orig[val])    # 标签的文字
            l.pack()    # 固定窗口位置
            labels.append(l)

def clear():
    for i in range(len(labels)):
        labels[i].destroy()
 

#window = tk.Toplevel()
window = tk.Tk()
window.title('Movie Recommendation')

tk.Label(window, text='Input an integer from 1 ~ 3952').pack()

e = tk.Entry(window)
e.pack()

b1 = tk.Button(window, text='Generate Recommendations',height=2,command=recommend_same_type_movie)
b1.pack()

b2 = tk.Button(window, text='Reset',height=2,command=clear)
b2.pack()

window.mainloop()

