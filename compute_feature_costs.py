
import time
from extract_features import *                                                  

def compute_feature_costs(candset, candset_l_key_attr, candset_r_key_attr,             
                          ltable, rtable, l_key_attr, r_key_attr, ft, sample_size):

    l_columns = list(ltable.columns.values)                                     
    l_key_attr_index = l_columns.index(l_key_attr)                              
    ltable_dict = build_dict_from_table(ltable, l_key_attr_index)               
                                                                                
    r_columns = list(rtable.columns.values)                                     
    r_key_attr_index = r_columns.index(r_key_attr)                              
    rtable_dict = build_dict_from_table(rtable, r_key_attr_index)               
                                                                                
    candset_columns = list(candset.columns.values)                              
    candset_l_key_attr_index = candset_columns.index(candset_l_key_attr)        
    candset_r_key_attr_index = candset_columns.index(candset_r_key_attr)        
                                                                                
    feat_cost = {}
    for feat_name in ft['feat_name']:
        feat_cost[feat_name] = 0.0
   
    sample_pairs = candset.sample(sample_size)

    for row in sample_pairs.itertuples(index = False):                         
        l_id = row[candset_l_key_attr_index]                                    
        r_id = row[candset_r_key_attr_index]                                    
                                                                                
        l_row = ltable_dict[l_id]                                               
        r_row = rtable_dict[r_id]                                               
                                                                                                                                     
        for feat_row in ft.itertuples(index = False):        
            if pd.isnull(l_row[feat_row[1]]) or pd.isnull(r_row[feat_row[2]]):  
                continue
            start_time = time.time()                                                        
            if feat_row[3] is None:                                             
                score = feat_row[4](l_row[feat_row[1]], r_row[feat_row[2]]) 
            else:                                                               
                score = feat_row[4](feat_row[3](l_row[feat_row[1]]),          
                                    feat_row[3](r_row[feat_row[2]]))             
            feat_cost[feat_row[0]] += (time.time() - start_time)

    f = open('feature_costs', 'w')                                              
    for feat_name in ft['feat_name']:
        feat_cost[feat_name] /= float(sample_size)
        f.write(feat_name + ',' + str(feat_cost[feat_name]) + '\n')
    f.close()
