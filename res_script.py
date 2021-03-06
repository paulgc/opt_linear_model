import pandas as pd
from autofeaturegen import get_features
from extract_features import *
import time

ldf=pd.read_csv('http://pages.cs.wisc.edu/~anhai/data/784_data/restaurants2/csv_files/zomato.csv', encoding='iso-8859-1')
rdf=pd.read_csv('http://pages.cs.wisc.edu/~anhai/data/784_data/restaurants2/csv_files/yelp.csv', encoding='iso-8859-1')
lab = pd.read_csv('/scratch/res2/labeled_data.csv')

lab = lab[['ltable.id', 'rtable.id', 'label']]
#lab = lab[['ltable.ID', 'rtable.ID']]
ft=get_features(ldf,rdf,set(['ID']), set(['ID']))

st = time.time()
#fvs = extract_fvs_candset(lab, 'ltable.ID', 'rtable.ID', ldf,rdf,'ID','ID',ft)
fvs = extract_fvs_labeled_pairs(lab, 'ltable.id', 'rtable.id', ldf,rdf,'ID','ID',ft, 'label')
fvs.to_csv('/scratch/res2/lab_fvs_v3.csv', index=False)
print('Time taken : ', time.time()-st)
