
from py_stringmatching.tokenizer.alphabetic_tokenizer import AlphabeticTokenizer
from py_stringmatching.tokenizer.alphanumeric_tokenizer import AlphanumericTokenizer
from py_stringmatching.tokenizer.qgram_tokenizer import QgramTokenizer
from py_stringmatching.tokenizer.whitespace_tokenizer import WhitespaceTokenizer
import pandas as pd 

from features import *


def get_features(ltable, rtable, l_exclude_attrs=set(), r_exclude_attrs=set()):

    toks_set = {'alph': AlphabeticTokenizer(return_set=True),                   
                'alph_num': AlphanumericTokenizer(return_set=True),             
                'ws': WhitespaceTokenizer(return_set=True),                     
                'qg2': QgramTokenizer(qval=2, return_set=True),                 
                'qg3': QgramTokenizer(qval=3, return_set=True)}                 
                                                                                
    toks_bag = {'alph_bag': AlphabeticTokenizer(return_set=False),              
                'alph_num_bag': AlphanumericTokenizer(return_set=False),        
                'ws_bag': WhitespaceTokenizer(return_set=False),                
                'qg2_bag': QgramTokenizer(qval=2, return_set=False),            
                'qg3_bag': QgramTokenizer(qval=3, return_set=False)}            
                                                                                
    str_features = {'jaccard': (jaccard, True, False),                          
                    'cosine': (cosine, True, False),                            
                    'dice': (dice, True, False),                                
                    'overlap_coeff': (overlap_coeff, True, False),              
                    'monge_elkan': (monge_elkan, True, False),                  
                    'tfidf': (tfidf, True, True),                               
                    'soft_tfidf': (soft_tfidf, True, True),                     
                    'lev_sim': (lev_sim, False),                                
#                    'hamming_sim': (hamming_sim, False),                        
                    'jaro': (jaro, False),                                      
                    'jaro_winkler': (jaro_winkler, False),                      
                    'needleman_wunsch': (needleman_wunsch, False),              
                    'smith_waterman': (smith_waterman, False),                  
                    'exact_match': (exact_match, False)}                        
                                                                                
    num_features = {'rel_diff': rel_diff,                                       
                    'abs_norm': abs_norm}  

    l_col_names = ltable.columns
    r_col_names = rtable.columns
    l_col_types = ltable.dtypes
    r_col_types = rtable.dtypes

    l_col_map = {}
    i = 0
    for l_col_name in l_col_names:
        if l_col_name in l_exclude_attrs:
            i += 1
            continue
        l_col_map[l_col_name] = (i, l_col_types[i])
        i += 1

    feat_rows = []        
    i = 0
    for r_col_name in r_col_names:
        if r_col_name in r_exclude_attrs:
            i += 1
            continue
        l_col = l_col_map.get(r_col_name) 

        if l_col is None:
            print('ERROR: Column ' + r_col_name + ' in  rtable not found in ltable')
            return
        if l_col[1] != r_col_types[i]:
            print('ERROR: Type mismatch for column ' + r_col_name + '. ' +\
                  r_col_types[i] + ' in rtable and ' + l_col[1] + ' in ltable.')

        if l_col[1] == int or l_col[1] == float:
            for k in num_features.keys(): 
                feat_rows.append((r_col_name + '_' + k, l_col[0], i, 
                                  None, num_features[k]))
        else:
            for k in str_features.keys():
                feat_entry = str_features[k] 
                if feat_entry[1] == False:                                      
                    feat_rows.append((r_col_name + '_' + k, l_col[0], i,      
                                      None, feat_entry[0]))
                else:
                    toks = toks_bag if feat_entry[2] else toks_set
                    for t in toks.keys():
                        feat_rows.append((r_col_name + '_' + k + '_' + t, 
                                          l_col[0], i,  
                                          toks[t].tokenize, feat_entry[0])) 
        i += 1

    feature_table = pd.DataFrame(feat_rows, 
                        columns=['feat_name', 'l_attr', 'r_attr', 'tok', 'sim_fn'])
    return feature_table

