from __future__ import division
import collections
from math import sqrt

from py_stringmatching.similarity_measure.jaro import Jaro
from py_stringmatching.utils import *

def soft_tfidf_norm(bag1, bag2):

    # if the strings match exactly return 1.0
    if sim_check_for_exact_match(bag1, bag2):
        return 1.0

    # if one of the strings is empty return 0
    if sim_check_for_empty(bag1, bag2):
        return 0

    sim_func = Jaro().get_raw_score
    threshold = 0.5

    # term frequency for input strings
    tf_x, tf_y = collections.Counter(bag1), collections.Counter(bag2)

    # find unique elements in the input lists and their document frequency 
    local_df = {}
    for element in tf_x:
        local_df[element] = local_df.get(element, 0) + 1
    for element in tf_y:
        local_df[element] = local_df.get(element, 0) + 1

    # if corpus is not provided treat input string as corpus
    curr_df, corpus_size = (local_df, 2)

    # calculating the term sim score against the input string 2,
    # construct similarity map
    similarity_map = {}
    for term_x in tf_x:
        max_score = 0.0
        for term_y in tf_y:
            score = sim_func(term_x, term_y)
            # adding sim only if it is above threshold and
            # highest for this element
            if score > threshold and score > max_score:
                similarity_map[term_x] = (score, term_x, term_y)
                max_score = score

    # position of first string, second string and sim score
    # in the tuple
    first_string_pos = 1
    second_string_pos = 2
    sim_score_pos = 0

    result, v_x_2, v_y_2 = 0.0, 0.0, 0.0
    # soft-tfidf calculation
    for element in local_df.keys():
        # denominator
        idf = corpus_size / curr_df[element]
        v_x = idf * tf_x.get(element, 0)
        v_x_2 += v_x * v_x
        v_y = idf * tf_y.get(element, 0)
        v_y_2 += v_y * v_y

    used_x = {}
    used_y = {}
    for sim in sorted(similarity_map.values(), reverse=True):
        if used_x.get(sim[first_string_pos]) is not None or used_y.get(sim[second_string_pos]) is not None:
            continue
        idf_first = corpus_size / curr_df.get(sim[first_string_pos], 1)     
        idf_second = corpus_size / curr_df.get(sim[second_string_pos], 1)   
        v_x = idf_first * tf_x.get(sim[first_string_pos], 0)                
        v_y = idf_second * tf_y.get(sim[second_string_pos], 0)              
        result += v_x * v_y * sim[sim_score_pos] 
        used_x[sim[first_string_pos]] = True
        used_y[sim[second_string_pos]]= True

    return result if v_x_2 == 0 else result / (sqrt(v_x_2) * sqrt(v_y_2))
