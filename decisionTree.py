# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 23:52:39 2018

@author: shuva
"""
import csv
import math
import sys

def split_data(data_list, attribute, value):
    result_list = list()
    for row in data_list:
        if row.get(attribute) == value:
            new_row = dict(row)
            result_list.append(new_row)
    return result_list

def calc_entropy(data_list,class_key, plus_value, split_attribute='', split_value=''):
    if split_attribute != '':
        sub_list = split_data(data_list,split_attribute,split_value)
    else :
        sub_list = data_list
    tot_count = len(sub_list)
    if tot_count == 0:
        return 0    
    plus_list = split_data(sub_list,class_key,plus_value)
    p1 = len(plus_list)/tot_count
    if p1 == 0 or p1 == 1 :
        Entropy = 0
    else :
        Entropy = p1*math.log(1/p1,2) + (1-p1)*math.log(1/(1-p1),2)
    return Entropy

def calc_mutual_info(data_list, class_key, plus_value, split_attribute):
    Entropy = calc_entropy(data_list, class_key, plus_value)
    tot_count = len(data_list)
    if tot_count == 0:
        return 0
    y_count = len(split_data(data_list,split_attribute, 'y'))
    entropy_y = calc_entropy(data_list, class_key, plus_value,
                             split_attribute=split_attribute,
                             split_value='y')
    entropy_n = calc_entropy(data_list, class_key, plus_value,
                             split_attribute=split_attribute,
                             split_value='n')
    mutual_info = Entropy - (y_count*entropy_y + (tot_count-y_count)*entropy_n)/tot_count
    #print('Entropy:', Entropy, 'y_entropy:', entropy_y, 'n_entropy:', entropy_n)
    return mutual_info

def get_split(data_list, class_key, plus_value):
    attributes = list(data_list[0].keys())
    best_attribute = attributes[0]
    best_mi = -1
    for attribute in attributes:
        if attribute == attributes[-1]:
            continue
        mi = calc_mutual_info(data_list,class_key, plus_value, attribute)
        if mi > best_mi and mi >= 0.1 : 
            best_mi = mi
            best_attribute = attribute
    #print(best_attribute, best_mi)
    return best_attribute, best_mi 

def print_split(data_list, key, plus_value,attribute_split=''):
    if attribute_split != '':
        for value in ['y', 'n']:
            plus_count = 0;
            tot_count  = 0;
            for row in data_list:
                if row[attribute_split] == value:
                    tot_count+=1
                    if row[key] == plus_value:
                        plus_count+=1
            print(attribute_split, ' = ', value, ': [',plus_count,tot_count-plus_count,  ']')

def run_decision_tree(data_list, class_key, plus_value, depth):
    decision_tree_list = list()
    best_attribute, best_mi = get_split(data_list, class_key, plus_value)
    #print('Running for depth :', depth, best_attribute, best_mi)
    if best_mi < 0.1  or depth > 2:
        return '',list()
    decision_tree_list.append(dict(root=best_attribute))
    y_split = split_data(data_list, best_attribute, 'y')
    n_split = split_data(data_list, best_attribute, 'n')
    for row in y_split:
        del row[best_attribute]
    for row in n_split:
        del row[best_attribute]
    if len(y_split) != 0:
        y_attribute, dummy = run_decision_tree(y_split,class_key, plus_value, depth+1 )
    else :
        y_attribute = ''
    if len(n_split) != 0:
        n_attribute, dummy = run_decision_tree(n_split,class_key, plus_value, depth+1 )
    else :
        n_attribute = ''
    if y_attribute!='':
        decision_tree_list.append(dict(y_root=y_attribute))
    if n_attribute!= '':
        decision_tree_list.append(dict(n_root= n_attribute))
    return best_attribute, decision_tree_list

def print_decision_tree(decision_tree_list, data_list, class_key, plus_value, mode='train'):
    index = 0
    root_attribute = decision_tree_list[index]['root']
    index+=1
    if index >= len(decision_tree_list):
        y_attribute = ''
    else :
        y_attribute = decision_tree_list[index].get('y_root', '')
    if y_attribute!='':
        index+=1
    if index >= len(decision_tree_list):
        n_attribute = ''
    else :
        n_attribute = decision_tree_list[index].get('n_root', '')
    error = 0;
    
    if mode == 'train':
        print('[', len(split_data(data_list,class_key, plus_value)), '+/',
               len(data_list) - len(split_data(data_list,class_key,plus_value)),'-]', sep='')
    
    y_split = split_data(data_list, root_attribute, 'y')
    if mode == 'train' :
        print(root_attribute, ' = y: ', '[', len(split_data(y_split,class_key, plus_value)), '+/',
          len(y_split) - len(split_data(y_split,class_key,plus_value)),'-]', sep='')
    
    if y_attribute != '':
        y_y_split = split_data(y_split, y_attribute, 'y')
        y_n_split = split_data(y_split, y_attribute, 'n')
        if mode == 'train':
            print('| ',y_attribute, ' = y: ', '[', len(split_data(y_y_split,class_key, plus_value)), '+/',
                                                len(y_y_split) - len(split_data(y_y_split,class_key,plus_value)),'-]', sep='')
       
            print('| ',y_attribute, ' = y: ', '[', len(split_data(y_n_split,class_key, plus_value)), '+/',
                                                len(y_n_split) - len(split_data(y_n_split,class_key,plus_value)),'-]', sep='')
        error+= min(len(split_data(y_y_split,class_key, plus_value)),
                 len(y_y_split) - len(split_data(y_y_split,class_key,plus_value)))
        error+= min(len(split_data(y_n_split,class_key, plus_value)),
                 len(y_n_split) - len(split_data(y_n_split,class_key,plus_value))) 
    else :
        error+= min(len(split_data(y_split,class_key, plus_value)),
                    len(y_split) - len(split_data(y_split,class_key,plus_value)) )
    
    n_split = split_data(data_list, root_attribute, 'n')
    if mode == 'train':
        print(root_attribute, ' = n: ', '[', len(split_data(n_split,class_key, plus_value)), '+/',
          len(n_split) - len(split_data(n_split,class_key,plus_value)),'-]', sep='')
    
    if n_attribute != '':
        n_y_split = split_data(n_split, n_attribute, 'y')
        n_n_split = split_data(n_split, n_attribute, 'n')
        if mode == 'train':
            print('| ',n_attribute, ' = y: ', '[', len(split_data(n_y_split,class_key, plus_value)), '+/',
                                                len(n_y_split) - len(split_data(n_y_split,class_key,plus_value)),'-]', sep='')
            print('| ',n_attribute, ' = n: ', '[', len(split_data(n_n_split,class_key, plus_value)), '+/',
                                                len(n_n_split) - len(split_data(n_n_split,class_key,plus_value)),'-]', sep='')    
        error+= min(len(split_data(n_y_split,class_key, plus_value)),
                 len(n_y_split) - len(split_data(n_y_split,class_key,plus_value)))
        error+= min(len(split_data(n_n_split,class_key, plus_value)),
                 len(n_n_split) - len(split_data(n_n_split,class_key,plus_value)))
    else :
        error+= min(len(split_data(n_split,class_key, plus_value)),
                    len(n_split) - len(split_data(n_split,class_key,plus_value)))
    if mode == 'train':
        print('error(train):',error/len(data_list))
    else :
        print('error(test):',error/len(data_list))


y_n_map = dict(A='y', notA='n', y='y', n='n', democrat='y', republican='n')
csv_file = open(sys.argv[1],mode='r')
csv_reader = csv.DictReader(csv_file)
data_list= list()
for row in csv_reader:
    for key in row:
        row[key] = y_n_map[row[key]]
    data_list.append(dict(row))
    attributes = list(row.keys())
    
for attribute in attributes:
    for row in data_list:
        row[attribute.replace(" ", "")] = row[attribute]

for row in data_list:
    for attribute in attributes:
        if attribute.replace(" ", "") != attribute:
            del row[attribute]

test_csv_file = open(sys.argv[2],mode='r')
test_csv_reader = csv.DictReader(test_csv_file)
test_data_list= list()
for row in test_csv_reader:
    for key in row:
        row[key] = y_n_map[row[key]]
    test_data_list.append(dict(row))
    
for attribute in attributes:
    for row in test_data_list:
        row[attribute.replace(" ", "")] = row[attribute]

for row in test_data_list:
    for attribute in attributes:
        if attribute.replace(" ", "") != attribute:
            del row[attribute]
            
attributes = list(data_list[0].keys())
class_key = attributes[-1]
plus_value = 'y'
#print_split(data_list,attributes[-1],'democrat',attribute_split='Anti_satellite_test_ban')

decision_tree_list = list()
root_attribute, decision_tree_list = run_decision_tree(data_list,class_key, plus_value,1)

#print(decision_tree_list)
print_decision_tree(decision_tree_list, data_list, class_key, plus_value)
print_decision_tree(decision_tree_list, test_data_list, class_key, plus_value,mode='test')



    