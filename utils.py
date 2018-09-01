from collections import Counter, defaultdict
from dataloader import Dictionary
import pickle
import numpy as np
import os
import sys
import time
import h5py

def load_prme_text_embeddings(path, N):
    def complement_embeddings(embeddings, N):
        tmp = np.zeros(shape=(N, embeddings.shape[1]-1))
        for i, emb in enumerate(embeddings):
            id = int(emb[0])
            tmp[id][:] = emb[1:]
        return tmp
    
    assert not path is None and not N is None    
    mat = np.loadtxt(path)
    print('loaded embeddings from {}, shape {}'.format(path, mat.shape))
    return complement_embeddings(mat, N)

def get_GE_embeddings(args):
    def complement_embeddings(embeddings, N):
        tmp = np.zeros(shape=(N, embeddings.shape[1]-1))
        for i, emb in enumerate(embeddings):
            id = int(emb[0])
            tmp[id][:] = emb[1:]
        return tmp

    embeddings = load_embedding_from_text(os.path.join('data', 'embeddings', args.pattern, '{}_no_word'.format(args.CITY), 'net_POI_vec.txt'))[:,1:]
    region_embeddings = complement_region_embeddings(load_embedding_from_text(os.path.join('data', 'embeddings', args.pattern, '{}_no_word'.format(args.CITY), 'net_reg_vec.txt')),N=100)
    time_embeddings = complement_embeddings(load_embedding_from_text(os.path.join('data', 'embeddings', args.pattern, '{}_no_word'.format(args.CITY), 'net_time_vec.txt')), args.n_timeslot)
    return embeddings, region_embeddings, time_embeddings

def get_ST_embeddings(args, path=None):
    path = os.path.join(args.ROOT, 'STSkipgram', 'benchmark'.format(args.CITY)) if path is None else path
    sem_embeds = load_embeddings(os.path.join(path, '{}_sem_emb.h5'.format(args.CITY)))
    embeddings = load_embeddings(os.path.join(path, '{}_embeddings.h5'.format(args.CITY)))
    time_embeddings = load_embeddings(os.path.join(path, '{}_time_embeddings.h5'.format(args.CITY)))
    return sem_embeds, embeddings, time_embeddings

def get_SK_wot_embeddings(args, path=None):
    path = os.path.join(args.ROOT, 'SkipGram', 'benchmark_{}_wo_time'.format(args.CITY)) if path is None else path
    embeddings = load_embeddings(os.path.join(path, '{}_embeddings.h5'.format(args.CITY)))
    return embeddings

def get_SK_wt_embeddings(args, path=None):
    path = os.path.join(args.ROOT, 'SkipGram', 'benchmark_{}_with_time'.format(args.CITY)) if path is None else path
    embeddings = load_embeddings(os.path.join(path, '{}_embeddings.h5'.format(args.CITY)))
    time_embeddings = load_embeddings(os.path.join(path, '{}_time.h5'.format(args.CITY)))
    return embeddings, time_embeddings

def get_PV_embeddings(args, path=None):
    path = os.path.join(args.ROOT, 'Baseline', 'Poi2vec', 'exp', 'no_weight') if path is None else path
    embeddings = load_embeddings(os.path.join(path, '{}_embeddings.h5'.format(args.CITY)))
    return embeddings

def decay_func(x, y):
    return abs((y-x).seconds)/60

def devide_train_and_test(trajs):
    train_set = list()
    test_set = list()
    for seq in trajs:
        lenseq = len(seq)
        train_set.append(seq[:int(lenseq*0.8)])
        tmptest = seq[int(lenseq*0.8):]
        if not len(tmptest) < 2:
            test_set.append(tmptest)
    return train_set, test_set

def save_sequences(args, seqs, path=None):
    path = os.path.join('data', '{}_seqs.pk'.format(args.CITY)) if path is None else path
    with open(path, 'wb') as f:
        pickle.dump(seqs, f)
    print('Saved sequences to {}'.format(path))
  
def load_poi2region(path):
    print('Loading data from {}'.format(path))
    with open(path, 'rb') as f:
        return pickle.load(f)
    
def load_embeddings(path):
    with h5py.File(path, 'r') as f:
        data = f.get('embeddings').value
    print('Reading dadta from {}'.format(path))
    return data

def digitize_datetime(dt, pattern):
    if pattern == 'day-6':
        bins = np.linspace(0,24,6+1)
        return np.digitize(list(map(lambda x:x.hour, dt)), bins)-1
    elif pattern == 'day-24':
        bins = np.linspace(0,24,24+1)
        return np.digitize(list(map(lambda x:x.hour, dt)), bins)-1
    elif pattern == 'weekly': #[0,6]
        bins = np.linspace(0,6,6+1)
        return np.digitize(list(map(lambda x:x.weekday(), dt)), bins)-1
    elif pattern == 'week-end':
        return np.array(list(map(lambda x:int(x.weekday()>4), dt)))
    elif pattern == 'hand': #[ 2.,  6., 10., 14., 18., 22., 26.])
        bins = np.linspace(0,24,6+1) + 2
        return np.digitize(list(map(lambda x:x.hour, dt)), bins) % 6
    else: return None

def load_embedding_from_text(path):
    converter = {0: lambda x: int(x.split('_')[-1])}
    mat = np.loadtxt(path, skiprows=1, converters=converter, encoding='ascii')
    print('loaded embeddings from {}, shape {}'.format(path, mat.shape))
    return mat

def get_sequences(origin_data, with_user=False):
    indices = [0,4] if not with_user else [0,1,4]
    grouped_data, sorted_key = group_data_by_id(origin_data)
    seqs = []
    for k, v in grouped_data.items():
        seqs.append(np.concatenate(v)[:,indices])
    return seqs

def group_data_by_id(data):
    res = defaultdict(list)
    for seq in data:
        res[seq[0][1]].append(seq)
    res.default_factory = None
    sorted_key = sorted(res.keys(), key=lambda x: len(res[x]), reverse=True)
    return res, sorted_key

def get_emb_from_ckpt(args):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    tick = time.time()
    
    graph = tf.Graph()
    with graph.as_default():
        model = STSkipgram(args)
        saver = tf.train.Saver(model.all_params)
        sess = tf.Session(config=config)
        sess = load_model_tf(saver, args, sess, path=os.path.join(args.LOG_DIR, 'best', 'model.ckpt'))
        sem_emb, full_emb, time_emb = sess.run([model.sem_emb, model.embeddings, model.time_embeddings])
    return sem_emb, full_emb, time_emb