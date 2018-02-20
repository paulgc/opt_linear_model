# coding=utf-8
"""
This module contains similarity functions supported by py_entitymatching
"""

import pandas as pd
import six

import py_stringmatching as sm
from nw_norm import nw_norm
from soft_tfidf_norm import soft_tfidf_norm

## String based similarity measures
def affine(s1, s2):
    if s1 is None or s2 is None:
        return pd.np.NaN
    if pd.isnull(s1) or pd.isnull(s2):
        return pd.np.NaN
    # Remove non-ascii characters. This will be fixed in the next version
    # if isinstance(s1, six.string_types):
    #     s1 = gh.remove_non_ascii(s1)
    # if isinstance(s2, six.string_types):
    #     s2 = gh.remove_non_ascii(s2)
    # Create the similarity measure object
    measure = sm.Affine()
    if not(isinstance(s1, six.string_types) or isinstance(s1, bytes)):
        s1 = str(s1)

    if not(isinstance(s2, six.string_types) or isinstance(s2, bytes)):
        s2 = str(s2)

    # Call the function to compute the similarity
    return measure.get_raw_score(s1, s2)

def hamming_dist(s1, s2):
    if s1 is None or s2 is None:
        return pd.np.NaN
    if pd.isnull(s1) or pd.isnull(s2):
        return pd.np.NaN
    # if isinstance(s1, six.string_types):
    #     s1 = gh.remove_non_ascii(s1)
    # if isinstance(s2, six.string_types):
    #     s2 = gh.remove_non_ascii(s2)
    # Create the similarity measure object
    measure = sm.HammingDistance()

    if not(isinstance(s1, six.string_types) or isinstance(s1, bytes)):
        s1 = str(s1)

    if not(isinstance(s2, six.string_types) or isinstance(s2, bytes)):
        s2 = str(s2)


    # Call the function to compute the distance
    return measure.get_raw_score(s1, s2)


def hamming_sim(s1, s2):
    if s1 is None or s2 is None:
        return pd.np.NaN
    if pd.isnull(s1) or pd.isnull(s2):
        return pd.np.NaN
    # if isinstance(s1, six.string_types):
    #     s1 = gh.remove_non_ascii(s1)
    # if isinstance(s2, six.string_types):
    #     s2 = gh.remove_non_ascii(s2)
    # Create the similarity measure object
    measure = sm.HammingDistance()
    if not(isinstance(s1, six.string_types) or isinstance(s1, bytes)):
        s1 = str(s1)

    if not(isinstance(s2, six.string_types) or isinstance(s2, bytes)):
        s2 = str(s2)

    # Call the function to compute the similarity score.
    return measure.get_sim_score(s1, s2)


def lev_dist(s1, s2):
    if s1 is None or s2 is None:
        return pd.np.NaN
    if pd.isnull(s1) or pd.isnull(s2):
        return pd.np.NaN
    # if isinstance(s1, six.string_types):
    #     s1 = gh.remove_non_ascii(s1)
    # if isinstance(s2, six.string_types):
    #     s2 = gh.remove_non_ascii(s2)
    # Create the similarity measure object
    measure = sm.Levenshtein()
    if not(isinstance(s1, six.string_types) or isinstance(s1, bytes)):
        s1 = str(s1)

    if not(isinstance(s2, six.string_types) or isinstance(s2, bytes)):
        s2 = str(s2)

    # Call the function to compute the distance measure.
    return measure.get_raw_score(s1, s2)


def lev_sim(s1, s2):
    if s1 is None or s2 is None:
        return pd.np.NaN
    if pd.isnull(s1) or pd.isnull(s2):
        return pd.np.NaN
    # if isinstance(s1, six.string_types):
    #     s1 = gh.remove_non_ascii(s1)
    # if isinstance(s2, six.string_types):
    #     s2 = gh.remove_non_ascii(s2)
    # Create the similarity measure object
    measure = sm.Levenshtein()
    if not(isinstance(s1, six.string_types) or isinstance(s1, bytes)):
        s1 = str(s1)

    if not(isinstance(s2, six.string_types) or isinstance(s2, bytes)):
        s2 = str(s2)

    # Call the function to compute the similarity measure
    return measure.get_sim_score(s1, s2)


def jaro(s1, s2):
    if s1 is None or s2 is None:
        return pd.np.NaN
    if pd.isnull(s1) or pd.isnull(s2):
        return pd.np.NaN
    # if isinstance(s1, six.string_types):
    #     s1 = gh.remove_non_ascii(s1)
    # if isinstance(s2, six.string_types):
    #     s2 = gh.remove_non_ascii(s2)
    # Create the similarity measure object
    measure = sm.Jaro()
    if not(isinstance(s1, six.string_types) or isinstance(s1, bytes)):
        s1 = str(s1)

    if not(isinstance(s2, six.string_types) or isinstance(s2, bytes)):
        s2 = str(s2)

    # Call the function to compute the similarity measure
    return measure.get_raw_score(s1, s2)


def jaro_winkler(s1, s2):
    if s1 is None or s2 is None:
        return pd.np.NaN
    if pd.isnull(s1) or pd.isnull(s2):
        return pd.np.NaN
    # if isinstance(s1, six.string_types):
    #     s1 = gh.remove_non_ascii(s1)
    # if isinstance(s2, six.string_types):
    #     s2 = gh.remove_non_ascii(s2)
    # Create the similarity measure object
    measure = sm.JaroWinkler()
    if not(isinstance(s1, six.string_types) or isinstance(s1, bytes)):
        s1 = str(s1)

    if not(isinstance(s2, six.string_types) or isinstance(s2, bytes)):
        s2 = str(s2)

    # Call the function to compute the similarity measure
    return measure.get_raw_score(s1, s2)


def needleman_wunsch(s1, s2):

    if s1 is None or s2 is None:
        return pd.np.NaN
    if pd.isnull(s1) or pd.isnull(s2):
        return pd.np.NaN
    # if isinstance(s1, six.string_types):
    #     s1 = gh.remove_non_ascii(s1)
    # if isinstance(s2, six.string_types):
    #     s2 = gh.remove_non_ascii(s2)
    # Create the similarity measure object
    measure = sm.NeedlemanWunsch()
    if not(isinstance(s1, six.string_types) or isinstance(s1, bytes)):
        s1 = str(s1)

    if not(isinstance(s2, six.string_types) or isinstance(s2, bytes)):
        s2 = str(s2)

    # Call the function to compute the similarity measure
    return nw_norm(s1, s2)


def smith_waterman(s1, s2):
    if s1 is None or s2 is None:
        return pd.np.NaN
    if pd.isnull(s1) or pd.isnull(s2):
        return pd.np.NaN
    # if isinstance(s1, six.string_types):
    #     s1 = gh.remove_non_ascii(s1)
    # if isinstance(s2, six.string_types):
    #     s2 = gh.remove_non_ascii(s2)
    # Create the similarity measure object
    measure = sm.SmithWaterman()
    if not(isinstance(s1, six.string_types) or isinstance(s1, bytes)):
        s1 = str(s1)

    if not(isinstance(s2, six.string_types) or isinstance(s2, bytes)):
        s2 = str(s2)

    # Call the function to compute the similarity measure
    return (measure.get_raw_score(s1, s2)/min(len(s1), len(s2)))


# Token-based measures
def jaccard(arr1, arr2):
    if arr1 is None or arr2 is None:
        return pd.np.NaN
    if not isinstance(arr1, list):
        arr1 = [arr1]
    if any(pd.isnull(arr1)):
        return pd.np.NaN
    if not isinstance(arr2, list):
        arr2 = [arr2]
    if any(pd.isnull(arr2)):
        return pd.np.NaN
    # Create jaccard measure object
    measure = sm.Jaccard()
    # Call a function to compute a similarity score
    return measure.get_raw_score(arr1, arr2)


def cosine(arr1, arr2):
    if arr1 is None or arr2 is None:
        return pd.np.NaN
    if not isinstance(arr1, list):
        arr1 = [arr1]
    if any(pd.isnull(arr1)):
        return pd.np.NaN
    if not isinstance(arr2, list):
        arr2 = [arr2]
    if any(pd.isnull(arr2)):
        return pd.np.NaN
    # Create cosine measure object
    measure = sm.Cosine()
    # Call the function to compute the cosine measure.
    return measure.get_raw_score(arr1, arr2)


def overlap_coeff(arr1, arr2):
    if arr1 is None or arr2 is None:
        return pd.np.NaN
    if not isinstance(arr1, list):
        arr1 = [arr1]
    if any(pd.isnull(arr1)):
        return pd.np.NaN
    if not isinstance(arr2, list):
        arr2 = [arr2]
    if any(pd.isnull(arr2)):
        return pd.np.NaN
    # Create overlap coefficient measure object
    measure = sm.OverlapCoefficient()
    # Call the function to return the overlap coefficient
    return measure.get_raw_score(arr1, arr2)

def dice(arr1, arr2):
    if arr1 is None or arr2 is None:
        return pd.np.NaN
    if not isinstance(arr1, list):
        arr1 = [arr1]
    if any(pd.isnull(arr1)):
        return pd.np.NaN
    if not isinstance(arr2, list):
        arr2 = [arr2]
    if any(pd.isnull(arr2)):
        return pd.np.NaN

    # Create Dice object
    measure = sm.Dice()
    # Call the function to return the dice score
    return measure.get_raw_score(arr1, arr2)

# Hybrid measure
def monge_elkan(arr1, arr2):
    if arr1 is None or arr2 is None:
        return pd.np.NaN
    if not isinstance(arr1, list):
        arr1 = [arr1]
    if any(pd.isnull(arr1)):
        return pd.np.NaN
    if not isinstance(arr2, list):
        arr2 = [arr2]
    if any(pd.isnull(arr2)):
        return pd.np.NaN
    # Create Monge-Elkan measure object
    measure = sm.MongeElkan()
    # Call the function to compute the Monge-Elkan measure
    return measure.get_raw_score(arr1, arr2)


def tfidf(arr1, arr2):                                                    
    if arr1 is None or arr2 is None:                                            
        return pd.np.NaN                                                        
    if not isinstance(arr1, list):                                              
        arr1 = [arr1]                                                           
    if any(pd.isnull(arr1)):                                                    
        return pd.np.NaN                                                        
    if not isinstance(arr2, list):                                              
        arr2 = [arr2]                                                           
    if any(pd.isnull(arr2)):                                                    
        return pd.np.NaN                                                        
    # Create TFIDF measure object                                         
    measure = sm.TfIdf()                                                   
    # Call the function to compute the TFIDF measure                      
    return measure.get_sim_score(arr1, arr2)    


def soft_tfidf(arr1, arr2):                                                          
    if arr1 is None or arr2 is None:                                            
        return pd.np.NaN                                                        
    if not isinstance(arr1, list):                                              
        arr1 = [arr1]                                                           
    if any(pd.isnull(arr1)):                                                    
        return pd.np.NaN                                                        
    if not isinstance(arr2, list):                                              
        arr2 = [arr2]                                                           
    if any(pd.isnull(arr2)):                                                    
        return pd.np.NaN                                                        
    # Create Soft TFIDF measure object                                               
    measure = sm.SoftTfIdf()                                                        
    # Call the function to compute the Soft TFIDF measure                      
    return soft_tfidf_norm(arr1, arr2)
#    return measure.get_raw_score(arr1, arr2)  


# boolean/string/numeric similarity measure
def exact_match(d1, d2):
    if d1 is None or d2 is None:
        return pd.np.NaN
    if pd.isnull(d1) or pd.isnull(d2):
        return pd.np.NaN
    # Check if they match exactly
    if d1 == d2:
        return 1
    else:
        return 0


# numeric similarity measure
def rel_diff(d1, d2):
    if d1 is None or d2 is None:
        return pd.np.NaN
    if pd.isnull(d1) or pd.isnull(d2):
        return pd.np.NaN
    d1 = float(d1)
    d2 = float(d2)
    if d1 == 0.0 and d2 == 0.0:
        return 0
    else:
        # Compute the relative difference between two numbers
        # ref: https://en.wikipedia.org/wiki/Relative_change_and_difference
#        x = (2*abs(d1 - d2)) / (d1 + d2)
        x = (abs(d1 - d2) / max(abs(d1), abs(d2)))                                        
        if x <= 10e-5:                                                          
            x = 0                                                               
        return 1.0 - x 
        return x


# compute absolute norm similarity
def abs_norm(d1, d2):
    if d1 is None or d2 is None:
        return pd.np.NaN
    if pd.isnull(d1) or pd.isnull(d2):
        return pd.np.NaN
    d1 = float(d1)
    d2 = float(d2)
    if d1 == 0.0 and d2 == 0.0:
        return 0
    else:
        # Compute absolute norm similarity between two numbers.
        x = (abs(d1 - d2) / max(d1, d2))
        if x <= 10e-5:
            x = 0
        return 1.0 - x
