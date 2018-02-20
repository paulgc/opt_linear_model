
import pandas as pd                                                             
from autofeaturegen import get_features                                         
import time                                                                     
from compute_feature_costs import *                                                
                                                                                
ldf=pd.read_csv('http://pages.cs.wisc.edu/~anhai/data/784_data/restaurants2/csv_files/zomato.csv', encoding='iso-8859-1')
rdf=pd.read_csv('http://pages.cs.wisc.edu/~anhai/data/784_data/restaurants2/csv_files/yelp.csv', encoding='iso-8859-1')
lab = pd.read_csv('/scratch/res2/candset.csv') 

lab = lab[['ltable.ID', 'rtable.ID']]                                           
ft=get_features(ldf,rdf,set(['ID']), set(['ID']))  

st = time.time()                                                                
compute_feature_costs(lab, 'ltable.ID', 'rtable.ID', ldf,rdf,'ID','ID', ft, 100)        
print('Time taken : ', time.time()-st)    
