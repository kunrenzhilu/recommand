import tensorflow as tf
import numpy as np
from tqdm import tqdm
from utils import digitize_datetime, decay_func
config = tf.ConfigProto()
config.gpu_options.allow_growth=True

def get_k_accuracy(res_mat):
    res_mat = np.array(res_mat)
    return [np.mean(np.sum(res_mat[:,:k], axis=1) > 0) for k in [1,5,10,15,20]]

class Recommander:
    def __init__(self, embeddings, time_embeddings=None, region_embeddings=None, sem_embeddings=None):
        emb = tf.constant(embeddings,dtype=tf.float32)
        time_emb = tf.constant(time_embeddings, dtype=tf.float32) if not time_embeddings is None else None 
        region_emb = tf.constant(region_embeddings, dtype=tf.float32) if not region_embeddings is None else None
        sem_emb = tf.constant(sem_embeddings, dtype=tf.float32) if not sem_embeddings is None else None
        
        self.decays = tf.placeholder(dtype=tf.float32, shape=(None))
        self.t_now = tf.placeholder(dtype=tf.int32, shape=(None))
        self.r_now = tf.placeholder(dtype=tf.int32, shape=(None))
        self.l_xs = tf.placeholder(dtype=tf.int32, shape=(None))
        
        self.user_profile = tf.reduce_sum(tf.nn.embedding_lookup(emb, self.l_xs) * tf.reshape(self.decays, (-1,1)), axis=0, keepdims=True)
        self.score = tf.matmul(self.user_profile, tf.transpose(emb))
        if not time_embeddings is None:
            self.time_emb_now = tf.nn.embedding_lookup(time_emb, self.t_now)
            if not sem_embeddings is None:
                self.score += tf.matmul(self.time_emb_now, tf.transpose(sem_emb))
            else:
                self.score += tf.matmul(self.time_emb_now, tf.transpose(emb))
        if not region_embeddings is None:
            self.r_emb_now = tf.nn.embedding_lookup(region_emb, self.r_now)
            self.score = tf.matmul(self.r_emb_now, tf.transpose(emb))
        self.score = tf.squeeze(self.score)
        
def get_recommandation_result(testset, args, poi2region, emb_set, mode):
    graph = tf.Graph()
    with graph.as_default():
        if mode == 'GE':
            embeddings, region_embeddings, time_embeddings = emb_set
            model = Recommander(embeddings, region_embeddings, time_embeddings)
        elif mode == 'STSG':
            sem_emb, embeddings, time_embeddings = emb_set
            model = Recommander(embeddings, time_embeddings=time_embeddings, sem_embeddings=sem_emb)
        elif mode == 'Skipgram_wt':
            embeddings, time_embeddings = emb_set
            model = Recommander(embeddings, time_embeddings=time_embeddings)
        elif mode == 'Skipgram_wot':
            embeddings = emb_set
            model = Recommander(embeddings)
        elif mode == 'POI2VEC':
            embeddings = emb_set
            model = Recommander(embeddings)

        sess = tf.Session(config=config)
        result_mat = list()
        for u, seq in tqdm(enumerate(testset), total=len(testset)):
            lenseq = len(seq)
            poiseq, dtseq = seq[:,0], seq[:,1]
            tslot_seq = digitize_datetime(dtseq, args.pattern)
            seq = np.concatenate([seq, digitize_datetime(seq[:,1],args.pattern)[:,None]], axis=1)

            for i in range(lenseq-1):
                history = seq[:i]
                target  = seq[i+1]        
                l_n, dt_n, tslot_n = seq[i] #now
                l_y, dt_y, tslot_y = target

                decays  = list()
                l_xs    = history[:,0].astype(int)
                for j, x in enumerate(history):
                    decays.append(decay_func(y=dt_y, x=x[1]))

                if mode == 'GE':
                    scores = sess.run(model.score, {model.l_xs:l_xs, model.decays:decays,
                                                     model.t_now: [tslot_n], model.r_now: [poi2region[l_n]]})
                elif mode == 'STSG':
                    scores = sess.run(model.score, {model.l_xs:l_xs, model.decays:decays,
                                                     model.t_now: [tslot_n]})
                elif mode == 'Skipgram_wt':
                    scores = sess.run(model.score, {model.l_xs:l_xs, model.decays:decays,
                                 model.t_now: [tslot_n]})
                elif mode == 'Skipgram_wot':
                    scores = sess.run(model.score, {model.l_xs:l_xs, model.decays:decays})
                elif mode == 'POI2VEC':
                    scores = sess.run(model.score, {model.l_xs:l_xs, model.decays:decays})
                result_mat.append((-scores).argsort()[:20] == l_y)
    return result_mat