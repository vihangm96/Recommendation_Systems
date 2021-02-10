#!/usr/bin/env python

import sys
import pdb
import math
import itertools
import collections
from pyspark import SparkContext

def process_line(line):
    tokens = [t for t in line.split(',') if t != '']
    if len(tokens) == 4 and tokens[2]=="5.0":
        return [tokens[:2]]
    else:
        return []

def get_baskets_python(filename):
    basket_name = None 
    basket = []
    baskets = []

    for line in open(filename, 'r', encoding='utf-8'):
        pair = process_line(line[:-1])
        if len(pair) == 0:
            continue
        else:
            user_id, movie_id = pair[0] #accommodate return format of flatMap

        if user_id != basket_name:
            if basket_name is not None:
                baskets.append((basket_name, basket))
            basket = [movie_id]
            basket_name = user_id
        else:
            basket.append(movie_id)

    baskets.append((basket_name, basket))

    return baskets

def inverse_dict(d):
    # {key: value} will become {value: key}
    return {v: k for k, v in d.items()}
def tuple_wrapper(s):
    if type(s) is not tuple:
        s = (s, )
    return s

def get_possible_k(item_dict, k):
    possible_k = {}
    for pair in itertools.combinations(item_dict.keys(), 2):
        pair_set = set()
        for i in range(2):
            pair_set = pair_set.union(tuple_wrapper(pair[i]))
        if len(pair_set) == k:
            possible_k[frozenset(pair_set)] = [pair[0], pair[1]]
    return possible_k

# baskets will be modified! So no need to have return value
def filter_basket(baskets, item_dict, k):
    if k == 2:
        possible_item = item_dict
    else:
        possible_item = set()
        possible_item = possible_item.union(*item_dict.keys())

    for i in range(len(baskets)):
        basket = baskets[i]
        items = basket[1]
        items_filterd = [item for item in items if item in possible_item]
        baskets[i] = (basket[0], items_filterd)

# Wrap in a function to be reused in Q3
def triangular_matrix_method(baskets, support, item_dict=None, k=2):
    if item_dict is None:
        item_dict = get_item_dict(baskets)  #item -> integer
    else:
        filter_basket(baskets, item_dict, k)

    item_dict_inv = inverse_dict(item_dict) #integer -> item. Inverse dict will be used when printing results
    n = len(item_dict)

    if k >= 3:
        possible_k = get_possible_k(item_dict, k)

    # Storage space is pre-allocated. Similiar to ArrayList
    # Convert 2D index to 1D index
    tri_matrix = [0] * (n * (n-1) // 2) # n * (n-1) always be even for n >= 2, use true division to make it a int

    # Key logic: Upper Triangular Matrix Method
    for basket in baskets:
        # Take a basket (user), iterate all items (artist)
        items = basket[1]

        # Equivalent to a double loop, but more concise
        for kpair in itertools.combinations(items, k):
            # kpair is a k element tuple, kpair[i] is item (string)
            if k >= 3:
                pair_set = frozenset(kpair)

                # Now kpair is a 2 element pair
                kpair = possible_k.get(pair_set, None)
                if kpair is None:
                    continue

            # i, j is integer index
            i = item_dict[kpair[0]]
            j = item_dict[kpair[1]]

            # Keep sorted in upper triangular order
            if i > j:
                j, i = i, j

            # Convert 2D index to 1D index
            idx = int((n*(n-1)/2) - (n-i)*((n-i)-1)/2 + j - i - 1)
            # Increase count by 1
            tri_matrix[idx] += 1

    # Extract results
    frequent_itemset_list = []
    for idx in range(len(tri_matrix)):
        # Convert 1D index to 2D index
        i = int(n - 2 - math.floor(math.sqrt(-8*idx + 4*n*(n-1)-7)/2.0 - 0.5))
        j = int(idx + i + 1 - n*(n-1)/2 + (n-i)*((n-i)-1)/2)

        count = tri_matrix[idx]
        item_i = item_dict_inv[i]
        item_j = item_dict_inv[j]

        # Keep sorted in ascii order. item_i, item_j are strings or tuple of strings
        # This implementation is ready for k>=3
        item_all = set()
        for item in (item_i, item_j):
            item_all = item_all.union(tuple_wrapper(item))

        item_all = tuple(sorted(list(item_all)))

        # apply support threshold
        if count >= support:
            frequent_itemset_list.append((item_all, count))

    # First sorted by the occurrence count in decreasing order
    # Then sort by ascii order of the first item, in ascending order
    # Then sort by ascii order of the second item, in ascending order
    frequent_itemset_list = sorted(frequent_itemset_list, key=lambda x: [-x[1]] + list(x[0]))
    return frequent_itemset_list

# Count occurrence of each item
def get_item_counter(baskets):
    item_counter = collections.Counter()
    for basket in baskets:
        items = basket[1]
        item_counter.update(items)
    return item_counter

# Assign an index for each item
def get_dict_from_frequent(frequent_list):
    item_dict = {}
    for item in frequent_list:
        item_dict[item] = len(item_dict)
    return item_dict

def apriori_all_method(baskets, support, item_counter, total_baskets=0):
    if type(baskets) is not list:
        baskets = list(baskets) #baskets are list now
    
    itemsets_1 = sorted([(k, v) for k, v in item_counter.items() if v >= support], key=lambda x: x[1], reverse=True)
    frequent_1 = [x[0] for x in itemsets_1]

    itemsets_list = [itemsets_1]
    frequent_list = frequent_1
    frequent_last = frequent_1

    k = 2
    while True:
        # get a dictionary of current frequent items
        # Note: only frequent item pairs from the last pass is needed
        item_dict = get_dict_from_frequent(frequent_last)

        # baskets will be modfied!
        itemsets = triangular_matrix_method(baskets, support, item_dict, k=k)
        if len(itemsets) > 0:
            frequent_last = [x[0] for x in itemsets]
            frequent_list += frequent_last
            itemsets_list.append(itemsets)
            k += 1
        else:
            break
    
    return itemsets_list

def get_frequent_dict(itemsets):
    freq_dict = {}
    for frequent_itemset in itemsets:
        freq_dict[frequent_itemset[0]] = frequent_itemset[1]
    return freq_dict

def get_frequent_dict_list(itemsets_list):
    frequent_dict_list = []
    
    n_itemsets = len(itemsets_list)
    
    for i in range(n_itemsets):
        frequent_dict_list.append(get_frequent_dict(itemsets_list[i]))
        
    return frequent_dict_list

def get_all_associations(baskets, support, interest):
    
    associations = []
    
    if type(baskets) is not list:
        baskets = list(baskets) 
    
    item_counter = get_item_counter(baskets)
    
    total_baskets = len(baskets)
    
    frequent_itemsets_list = apriori_all_method(baskets, support, item_counter)
    
    frequent_dict_list = get_frequent_dict_list(frequent_itemsets_list)
    
    n_itemsets = len(frequent_dict_list)
    
    for item_j, support_j in item_counter.items():
        
        if support_j < support:
            continue
        
        for i in range(n_itemsets-1):

            for itemset_i, support_i in frequent_dict_list[i].items():

                if type(itemset_i) == str and item_j == itemset_i:
                    continue
                
                if item_j in itemset_i:
                    continue                
                
                if(type(itemset_i)==str):
                    union = [itemset_i]
                else:
                    union = list(itemset_i)
                union.append(item_j)
                union.sort()
                union = tuple(union)

                support_iUj = 0

                if union in frequent_dict_list[i+1]:
                    support_iUj = frequent_dict_list[i+1][union]
                                                                    
                conf_ij = support_iUj/support_i
                pr_j = support_j/total_baskets
                interest_ij = conf_ij - pr_j
                
                if interest_ij >= interest and support_iUj >= support:
                    if type(itemset_i)==str:
                        association = [[int(itemset_i)], int(item_j), interest_ij, support_iUj]
                    else:
                        itemset_list = []
                        for item in itemset_i:
                            itemset_list.append(int(item))
                        itemset_list.sort()
                        association = [itemset_list, int(item_j), interest_ij, support_iUj]
                    associations.append(association)
    
    associations.sort(key = lambda x: (-x[2], -x[3], x[0], x[1]))
    
    return associations

if __name__ == '__main__':

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    interest = float(sys.argv[3])
    support = int(sys.argv[4])
    
    #input_file = 'ml-latest-small/ratings_test_truth.csv'
    #output_file = 'task1_output.json'
    #interest = 0.2
    #support = 2
    
    baskets = get_baskets_python(input_file)
    
    associations = get_all_associations(baskets, support, interest)
    
    file_writer = open(output_file, 'w') 
    file_writer.write(str(associations)) 
    file_writer.close() 