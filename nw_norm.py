
import numpy as np
from py_stringmatching.utils import *

def sim_ident(char1, char2):
    return 0.0 if char1 == char2 else -1.0

def nw_norm(string1, string2):
    string1 = convert_to_unicode(string1)                                       
    string2 = convert_to_unicode(string2)     
    n, m = len(string1), len(string2)                                           

    if n==0 and m==0:
        return 1.0
    
    dist = nw_dist(string1, string2, n, m)
    return (-1 * dist + 2 * max(n,m)) / (2 * max(n,m))
    
def nw_dist(string1, string2, n, m):
    if string1 == string2:
        return 0
 
    gap_cost = -2.0
    if n == 0:
        return -1 * gap_cost * m
    if m == 0:
        return -1 * gap_cost * n
  
    dist_mat = np.zeros((len(string1) + 1, len(string2) + 1),
                        dtype=np.float)

    v0 = np.zeros(m+1, dtype=np.float)
    v1 = np.zeros(m+1, dtype=np.float)

    for j in xrange(m+1):
        v0[j]=j

    for i in xrange(1, n+1):
        v1[0] = i
        for j in xrange(1, m+1):
            v1[j] = min(v0[j] - gap_cost,
                        v1[j-1] - gap_cost,
                        v0[j-1] - sim_ident(string1[i-1], string2[j-1]))    

        v0, v1 = v1, v0

    return v0[m] 

