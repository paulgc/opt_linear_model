import pandas as pd


def build_dict_from_table(table, key_attr_index):
    table_dict = {}
    for row in table.itertuples(index=False):
        table_dict[row[key_attr_index]] = tuple(row)
    return table_dict

def extract_fvs_labeled_pairs(labeled_pairs, candset_l_key_attr, candset_r_key_attr,
                              ltable, rtable, l_key_attr, r_key_attr, ft, 
                              label_attr='label'):
    l_columns = list(ltable.columns.values)
    l_key_attr_index = l_columns.index(l_key_attr)
    ltable_dict = build_dict_from_table(ltable, l_key_attr_index)

    r_columns = list(rtable.columns.values)                                     
    r_key_attr_index = r_columns.index(r_key_attr)   
    rtable_dict = build_dict_from_table(rtable, r_key_attr_index)               

    candset_columns = list(labeled_pairs.columns.values)
    candset_l_key_attr_index = candset_columns.index(candset_l_key_attr)
    candset_r_key_attr_index = candset_columns.index(candset_r_key_attr)

    label_attr_index = candset_columns.index(label_attr)

    fvs = []
    for row in labeled_pairs.itertuples(index = False):
        l_id = row[candset_l_key_attr_index]
        r_id = row[candset_r_key_attr_index]

        l_row = ltable_dict[l_id]
        r_row = rtable_dict[r_id]

        fv = [l_id, r_id]
        for feat_row in ft.itertuples(index = False):
            if pd.isnull(l_row[feat_row[1]]) or pd.isnull(r_row[feat_row[2]]):
                fv.append(0.0)
                continue
            if feat_row[3] is None:
                fv.append(feat_row[4](l_row[feat_row[1]], r_row[feat_row[2]]))
            else:
                fv.append(feat_row[4](feat_row[3](l_row[feat_row[1]]), 
                                      feat_row[3](r_row[feat_row[2]])))
        fv.append(row[label_attr_index])

        fvs.append(fv)

    header = [candset_l_key_attr, candset_r_key_attr]
    header.extend(list(ft['feat_name'].values))
    header.append(label_attr)

    fvs_table = pd.DataFrame(fvs, columns = header)
    return fvs_table 


def extract_fvs_candset(candset, candset_l_key_attr, candset_r_key_attr,
                        ltable, rtable, l_key_attr, r_key_attr, ft):                              
    l_columns = list(ltable.columns.values)                                     
    l_key_attr_index = l_columns.index(l_key_attr)                              
    ltable_dict = build_dict_from_table(ltable, l_key_attr_index)               
                                                                                
    r_columns = list(rtable.columns.values)                                     
    r_key_attr_index = r_columns.index(r_key_attr)                              
    rtable_dict = build_dict_from_table(rtable, r_key_attr_index)               
                                                                                
    candset_columns = list(candset.columns.values)                        
    candset_l_key_attr_index = candset_columns.index(candset_l_key_attr)        
    candset_r_key_attr_index = candset_columns.index(candset_r_key_attr)        
                                                                                
    fvs = []                                                                    
    for row in candset.itertuples(index = False):                         
        l_id = row[candset_l_key_attr_index]                                    
        r_id = row[candset_r_key_attr_index]                                    
                                                                                
        l_row = ltable_dict[l_id]                                               
        r_row = rtable_dict[r_id]                                               
                                                                                
        fv = [l_id, r_id]                                                       
        for feat_row in ft.itertuples(index = False):                           
            if pd.isnull(l_row[feat_row[1]]) or pd.isnull(r_row[feat_row[2]]):  
                fv.append(0.0)                                                  
                continue                                                        
            if feat_row[3] is None:                                             
                fv.append(feat_row[4](l_row[feat_row[1]], r_row[feat_row[2]]))  
            else:                                                               
                fv.append(feat_row[4](feat_row[3](l_row[feat_row[1]]),          
                                      feat_row[3](r_row[feat_row[2]])))         
                                                                                
        fvs.append(fv)                                                          
                                                                                
    header = [candset_l_key_attr, candset_r_key_attr]                           
    header.extend(list(ft['feat_name'].values))                                 
                                                                                
    fvs_table = pd.DataFrame(fvs, columns = header)                             
    return fvs_table                   
