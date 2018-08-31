from recommand_np import get_recommandation_result, get_k_accuracy
import multiprocessing as mp
import time
import numpy as np

def map_get_recommandation(args):
    testset, args, poi2region, emb_set, mode = args
    return get_recommandation_result(testset, args, poi2region, emb_set, mode)

def reduce_recommendation_results(results):
    return get_k_accuracy(np.concatenate(results))

def multiprocess_recommand(nProcess, testdata, args, poi2region, emb_set, mode):
    try:
        tick = time.time()
        pool = mp.Pool(processes=nProcess)
        chunks = np.array_split(testdata, nProcess)
        results = pool.map(map_get_recommandation, zip(chunks,  [args]*nProcess, [poi2region]*nProcess, [emb_set]*nProcess, [mode]*nProcess))
        score = reduce_recommendation_results(results)
        pool.close()
        pool.join()
        print('Job Done, used time {}'.format(time.time()-tick))
        return score
    except:
        print('Error')
        pool.close()
        pool.join()
        raise