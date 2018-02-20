
from extract_features import *


def execute_model1(candset, candset_l_key_attr, candset_r_key_attr,
                   ltable, rtable, l_key_attr, r_key_attr, model, ft):

    l_columns = list(ltable.columns.values)                                     
    l_key_attr_index = l_columns.index(l_key_attr)                              
    ltable_dict = build_dict_from_table(ltable, l_key_attr_index)               
                                                                                
    r_columns = list(rtable.columns.values)                                     
    r_key_attr_index = r_columns.index(r_key_attr)                              
    rtable_dict = build_dict_from_table(rtable, r_key_attr_index)               
                                                                                
    candset_columns = list(candset.columns.values)                        
    candset_l_key_attr_index = candset_columns.index(candset_l_key_attr)        
    candset_r_key_attr_index = candset_columns.index(candset_r_key_attr)        
    
    feature_attrs = list(ft['feat_name'])
    
    coeff = list(model.coef_[0])
    intercept = model.intercept_[0] 
    n = len(coeff)
    print('num coefficients: ', n)
   
    ft = ft.set_index('feat_name') 
    feats_map = {}
    for i in xrange(n):
        feats_map[i] = ft.ix[feature_attrs[i]].values

    order = sorted([(abs(coeff[i]), coeff[i], i) for i in xrange(n)], reverse=True)

    pos_upper_bound = sum(map(lambda e: e[0] if e[1]>0 else 0, order))
    neg_upper_bound = sum(map(lambda e: e[0] if e[1]<0 else 0, order))                                                                
    print(pos_upper_bound, neg_upper_bound)

    labels = []                                                               
    for row in candset.itertuples(index = False):
        l_id = row[candset_l_key_attr_index]                                    
        r_id = row[candset_r_key_attr_index]                                    
                                                                                
        l_row = ltable_dict[l_id]                                               
        r_row = rtable_dict[r_id] 
        pub = pos_upper_bound
        nub = neg_upper_bound
        score = intercept

        for entry in order:
            if score - nub >= 0:
                break
            if score + pub < 0:
                break
            score += entry[1]*compute_feature(l_row, r_row, feats_map[entry[2]])
            if entry[1] > 0:
                pub -= entry[1]
            else:
                nub -= entry[0]    
        if score - nub >= 0:
            labels.append(1)
        else:
            labels.append(0)

    print(sum(labels))
    gold = candset[[candset_l_key_attr, candset_r_key_attr]]
    gold['label'] = labels
    gold.to_csv('/scratch/res2/pred_v5.csv', index=False)                


def compute_feature(l_row, r_row, feat_row):
    if pd.isnull(l_row[feat_row[0]]) or pd.isnull(r_row[feat_row[1]]):  
        return 0.0                                                  
                                                            
    if feat_row[2] is None:                                             
        return feat_row[3](l_row[feat_row[0]], r_row[feat_row[1]])  
    else:                                                               
        return feat_row[3](feat_row[2](l_row[feat_row[0]]),          
                           feat_row[2](r_row[feat_row[1]]))

def gen_ordering_by_coeffs(coeff, n):
    coeffs_tuple = [(abs(coeff[i]), coeff[i], i) for i in xrange(n)]
    neg_coeffs = []
    pos_coeffs = []
    for entry in coeffs_tuple:
        if entry[1] < 0:
            neg_coeffs.append(entry)
        elif entry[1] > 0:
            pos_coeffs.append(entry)
    return (neg_coeffs, sorted(pos_coeffs, reverse=True))
        
