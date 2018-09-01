import numpy as np
from tqdm import tqdm
from utils import digitize_datetime, decay_func

def get_k_accuracy(res_mat):
    res_mat = np.array(res_mat)
    return [np.mean(np.sum(res_mat[:,:k], axis=1) > 0) for k in [1,5,10,15,20]]

def get_recommandation_result(testset, args, poi2region, emb_set, mode, weight=[1,1,1]):
    if mode == 'GE':
        embeddings, region_embeddings, time_embeddings = emb_set
    elif mode == 'STSG':
#         sem_emb, embeddings, time_embeddings = emb_set
        sem_emb, geo_emb, time_embeddings = emb_set
        sem_dim = sem_emb.shape[1]; geo_dim = geo_emb.shape[1];
        embeddings = np.concatenate([sem_emb, geo_emb], axis=1)
    elif mode == 'Skipgram_wt':
        embeddings, time_embeddings = emb_set
    elif mode == 'Skipgram_wot':
        embeddings = emb_set
    elif mode == 'POI2VEC':
        embeddings = emb_set
    elif mode == 'PRME':
        embeddings, embeddings_u, user_embeddings = emb_set

    result_mat = list()
    for u, seq in tqdm(enumerate(testset), total=len(testset)):
        lenseq = len(seq)
        lenitem = len(seq[0])
        poiseq, dtseq = seq[:,0], seq[:,lenitem-1]
        tslot_seq = digitize_datetime(dtseq, args.pattern)
        seq = np.concatenate([seq, digitize_datetime(dtseq, args.pattern)[:,None]], axis=1)

        for i in range(lenseq-1):
            history = seq[:i]
            target  = seq[i+1]        
            if lenitem == 2:
                l_n, dt_n, tslot_n = seq[i] #now
                l_y, dt_y, tslot_y = target
            elif lenitem == 3:
                l_n, usr, dt_n, tslot_n = seq[i]
                l_y, usr, dt_y, tslot_y = target
            else: assert 0
                
            decays  = list()
            l_xs    = history[:,0].astype(int)
            for j, x in enumerate(history):
                decays.append(decay_func(y=dt_y, x=x[lenitem-1]))

            embs = embeddings[l_xs]
            if args.use_decay == True:
                user_profile = np.sum(embs*np.array(decays).reshape(-1,1), axis=0)
            else: 
                user_profile = np.mean(embs, axis=0)
            if not mode in ['POI2VEC', 'Skipgram_wot', 'PRME']:
                t_emb_now = time_embeddings[tslot_n]

            if mode == 'GE':
                r_emb_now = region_embeddings[poi2region[l_n]]
                scores = np.matmul(user_profile, embeddings.T) + \
                    np.matmul(r_emb_now, embeddings.T) + \
                    np.matmul(t_emb_now, embeddings.T)
            elif mode == 'STSG':
#                 scores = np.matmul(user_profile, embeddings.T) + \
#                     np.matmul(t_emb_now, sem_emb.T)
  
                profile_sem, profile_geo = user_profile[:sem_dim], user_profile[sem_dim:]
                score_sem = np.matmul(profile_sem, sem_emb.T) / float(sem_dim)
                score_geo = np.matmul(profile_geo, geo_emb.T) / float(geo_dim)
                score_time = np.matmul(t_emb_now, sem_emb.T) / float(sem_dim)
                scores = np.stack([score_sem, score_geo, score_time],axis=0)*np.array(weight).reshape(3,1)
                scores = np.sum(scores, axis=0)
            elif mode == 'Skipgram_wt':
                scores = np.matmul(user_profile, embeddings.T) + \
                    np.matmul(t_emb_now, embeddings.T)
            elif mode == 'Skipgram_wot':
                scores = np.matmul(user_profile, embeddings.T)
            elif mode == 'POI2VEC':
                scores = np.matmul(user_profile, embeddings.T)
            elif mode == 'PRME':
                user_profile = user_embeddings[usr]
                scores = np.matmul(user_profile, embeddings_u.T) + \
                    np.matmul(embeddings[l_n], embeddings.T)
#                 scores = np.matmul(user_profile, embeddings.T)
            result_mat.append((-scores).argsort()[:20] == l_y)
    return result_mat