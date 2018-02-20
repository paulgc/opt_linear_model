import pandas as pd
from autofeaturegen import get_features
from extract_features import *
import time
#from opt_execute_model1 import * 
from opt_execute_model_abs import *

ldf=pd.read_csv('http://pages.cs.wisc.edu/~anhai/data/784_data/restaurants2/csv_files/zomato.csv', encoding='iso-8859-1')
rdf=pd.read_csv('http://pages.cs.wisc.edu/~anhai/data/784_data/restaurants2/csv_files/yelp.csv', encoding='iso-8859-1')
lab = pd.read_csv('/scratch/res2/candset.csv')

#lab = lab[['ltable.id', 'rtable.id', 'label']]
lab = lab[['ltable.ID', 'rtable.ID']]
ft=get_features(ldf,rdf,set(['ID']), set(['ID']))

import pickle
lr = pickle.load(open('/scratch/res2/logreg_model_v3', 'r'))

st = time.time()
execute_model1(lab, 'ltable.ID', 'rtable.ID', ldf,rdf,'ID','ID', lr, ft)
#fvs = extract_fvs_labeled_pairs(lab, 'ltable.id', 'rtable.id', ldf,rdf,'ID','ID',ft, 'label')
print('Time taken : ', time.time()-st)
