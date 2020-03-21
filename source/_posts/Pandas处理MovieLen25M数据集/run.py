from mpi4py import MPI
import pickle as pkl
import pandas as pd
import numpy as np
from random import choice
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def get_dense(user_id, item_id, genome_feature, genres_tag_one_hot_feature, movie_rating_feature, tag_feature_movie, tag_feature_user, user_rating_feature):
    instance = []
    
    u_tag_ft = tag_feature_user[tag_feature_user['userId']==user_id].values
    if u_tag_ft.shape[0]==0:
        u_tag_ft = [0]*184
    else:
        u_tag_ft = u_tag_ft[0,1:]
    instance.extend(list(u_tag_ft))

    u_rate_ft = user_rating_feature[user_rating_feature['userId']==user_id].values
    if u_rate_ft.shape[0]==0:
        u_rate_ft = [0,0]
    else:
        u_rate_ft = u_rate_ft[0,1:]
    instance.extend(list(u_rate_ft))

    ## item feature
    i_gtag_ft = genres_tag_one_hot_feature[genres_tag_one_hot_feature['movieId']==item_id].values
    if i_gtag_ft.shape[0]==0:
        i_gtag_ft = [0]*20
    else:
        i_gtag_ft = i_gtag_ft[0,1:]
    instance.extend(list(i_gtag_ft))

    i_tag_ft = tag_feature_movie[tag_feature_movie['movieId']==item_id].values
    if i_tag_ft.shape[0]==0:
        i_tag_ft = [0]*184
    else:
        i_tag_ft = i_tag_ft[0,1:]
    instance.extend(list(i_tag_ft))

    i_relevence_ft = genome_feature[genome_feature['movieId']==item_id].values
    if i_relevence_ft.shape[0]==0:
        i_relevence_ft = [0]*1128
    else:
        i_relevence_ft = [float(x) for x in i_relevence_ft[0,1].split(',')]
    instance.extend(list(i_relevence_ft))

    i_rate_ft = movie_rating_feature[movie_rating_feature['movieId']==item_id].values
    if i_rate_ft.shape[0]==0:
        i_rate_ft = [0,0]
    else:
        i_rate_ft = i_rate_ft[0,1:]
    instance.extend(list(i_rate_ft))
    
    return instance

def generate_dataset(viewed_sequence, item_features, user_features, movie_set):
    genome_feature, genres_tag_one_hot_feature, movie_rating_feature, tag_feature_movie = item_features
    tag_feature_user, user_rating_feature = user_features
    
    f1 = open('train_{}.txt'.format(rank),'w')
    f2 = open('test_{}.txt'.format(rank),'w')

    cnt = 0
    userId_lst = viewed_sequence['userId'].values
    seq_lst = viewed_sequence.iloc[:,1:].values
    for i, user_id in enumerate(userId_lst):
        if i%size != rank:
            continue
        cnt += 1
        full_seq = seq_lst[0,0]
        item_lst = [int(x) for x in full_seq.split(',')]
        for j, item_id in enumerate(item_lst[::-1]):
            if j > 5:
                break

            instance = [user_id, item_id, 1]
            
            ## seq feature
            seq = full_seq.split(str(item_id))[0][:-1]
            if seq.count(',') < 10:
                break
            seq = '*'.join(seq.split(',')[-10:])
            instance.append(seq)
            
            ## dense & one-hot feature:
            dense_ft = get_dense(int(user_id), int(item_id), genome_feature, genres_tag_one_hot_feature, movie_rating_feature, tag_feature_movie, tag_feature_user, user_rating_feature)
            instance.extend(dense_ft)
            
            ## add to dataset
            if j==0:
                for each in instance:
                    f2.write(str(each)+',')
                f2.write('\n')
            else:
                for each in instance:
                    f1.write(str(each)+',')
                f1.write('\n')
            
            ## add a neg-instance
            while True:
                neg_item_id = choice(movie_set)
                if neg_item_id not in item_lst:
                    break
            neg_inst = [user_id, neg_item_id, 0, seq]
            neg_dense_ft = get_dense(user_id, neg_item_id, genome_feature, genres_tag_one_hot_feature, movie_rating_feature, tag_feature_movie, tag_feature_user, user_rating_feature)
            neg_inst.extend(neg_dense_ft)
            
            for each in neg_inst:
                f1.write(str(each)+',')
            f1.write('\n')
        
        if cnt % 100==0:
            print('worker {}, iter={}'.format(rank,cnt))
        # if i%20==0:
        #     print('processing: {}/{}, used time: {}, remains: {}/{}'.format(i,num_iter, sec_to_str(used_time), sec_to_str(avg_eta), sec_to_str(crt_eta)))
    f1.close()
    f2.close()
    return True

f = open('processed_ft.pkl','rb')
viewed_sequence,genome_feature, genres_tag_one_hot_feature, movie_rating_feature, tag_feature_movie,tag_feature_user, user_rating_feature = pkl.load(f)
f.close()

f = open('movie_set.pkl','rb')
movie_set = pkl.load(f)
f.close()

item_features = [genome_feature, genres_tag_one_hot_feature, movie_rating_feature, tag_feature_movie]
user_features = [tag_feature_user, user_rating_feature]
flag = generate_dataset(viewed_sequence, item_features, user_features, movie_set)

# f = open('data{}.pkl'.format(rank), 'wb')
# pkl.dump([train_data, test_data], f, 2)
# f.close()