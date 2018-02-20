
from extract_features import *
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

def cluster_features(candset, candset_l_key_attr, candset_r_key_attr,        
                     ltable, rtable, l_key_attr, r_key_attr, ft, sample_size, threshold):
    sample_candset = candset.sample(sample_size)

    sample_fvs = extract_fvs_candset(sample_candset, candset_l_key_attr, candset_r_key_attr,        
                                     ltable, rtable, l_key_attr, r_key_attr, ft)
    feature_attrs = list(sample_fvs.columns)
    feature_attrs.remove(candset_l_key_attr)
    feature_attrs.remove(candset_r_key_attr)
    print(feature_attrs) 
    print(len(feature_attrs))
    Y = pdist(sample_fvs[feature_attrs].fillna(0.0).transpose(), 'correlation')
    Z = linkage(Y)

    clusters = {}
    n = 0
    for i in xrange(len(feature_attrs)):
        clusters[i] = set([i])
        n += 1

    for i in xrange(len(Z)):
        if Z[i][2] > threshold:
            break 
        clusters[n] = set()
        clusters[n].update(clusters[int(Z[i][0])])
        clusters[n].update(clusters[int(Z[i][1])])
        del clusters[int(Z[i][0])]
        del clusters[int(Z[i][1])]                              
        n += 1

    return (clusters, Z, Y)
