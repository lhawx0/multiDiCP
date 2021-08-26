from copy import deepcopy
from copy import copy
from concurrent.futures import ThreadPoolExecutor, process
from functools import partial
from operator import itemgetter
from random import choice
from random import shuffle
from re import S
from networkx.algorithms.centrality import group
from networkx.classes.function import degree, neighbors, number_of_edges, number_of_nodes, selfloop_edges
from numpy.core.numeric import Inf
from numpy.random import randint
from numpy.random import lognormal
from collections import Counter
from networkx.generators.degree_seq import expected_degree_graph
import multiprocessing
import networkx as nx
import matplotlib.pyplot as plt
import math
import csv
import cpalgorithm as cp
import numpy as np
import json
import pandas as pd
import parmap
import matplotlib as mpl
from numba import njit
from numba import jit
class NetNode():
    def __init__(self,index,group,partition) -> None:
        self.index = index
        self.group = group
        self.partition = partition
    
class Group():
    def __init__(self) -> None:
        pass

def calculate_Q(adjM,partition):
    hit = 0
    num_Cin = np.count_nonzero(partition == 1 )
    num_Cout = np.count_nonzero(partition == 2 )
    num_Pin = np.count_nonzero(partition == 3 )
    num_Pout = np.count_nonzero(partition == 4)
    number_nodes = len(partition)


    #partition(1 = Cin 2=Cout 3=Pin 4=Pout)
    for i in range(0,number_nodes):
        if partition[i] == 1:
            for j in range(0,number_nodes):
                if(j != i) and (partition[j] == 1) and(adjM[i][j]!=0):
                    hit += 1
                if(j !=i) and (partition[j] == 4 or partition[j] == 3) and(adjM[i][j]!=0):
                    hit -= 0.5
                #if(j != i) and (partition[j] == 2) and(adjM[i][j]!=0):
                #   hit -= 1
        elif partition[i] == 2:
            for j in range(0,number_nodes):
                if(j != i) and (partition[j] !=4 ) and(adjM[i][j]!=0):
                    hit += 1
                if(j !=i) and (partition[j] == 4) and(adjM[i][j]!=0):
                    hit -= 0.5
        elif partition[i] == 3:
            for j in range(0,number_nodes):
                if (j != i ) and (partition[j] == 1 or partition[j] == 2) and (adjM[i][j]!=0):
                    hit -= 0.5
                elif (j != i ) and (partition[j] == 3 or partition[j] == 4) and (adjM[i][j]!=0):
                    hit -= 0.5
        elif partition[i] == 4:
            for j in range(0,number_nodes):
                if(j != i) and (partition[j] == 1) and(adjM[i][j]!=0):
                    hit += 1
                if (j != i ) and (partition[j] ==2) and (adjM[i][j]!=0):
                    hit -= 0.5
                if (j != i ) and (partition[j] == 3 or partition[j] == 4) and (adjM[i][j]!=0):
                    hit -= 0.5

    max_hit = num_Cin*(num_Cin - 1) + num_Cout *(num_Cout - 1 + num_Cin +num_Pin) + num_Pout*num_Cin
    Q = hit/max_hit
    
    return Q

#@njit
def calculate_Q_M2(adjM,partition_old,group_old,target,par_new,group_new):
    partition = np.copy(partition_old)
    group = np.copy(group_old)
    partition[target],group[target] = par_new,group_new
    hit = 0
    num_Cin = np.count_nonzero(partition == 1 )
    num_Cout = np.count_nonzero(partition == 2 )
    num_Pin = np.count_nonzero(partition == 3 )
    num_Pout = np.count_nonzero(partition == 4)
    number_nodes = len(partition)


    #partition(1 = Cin 2=Cout 3=Pin 4=Pout)
    for i in range(0,number_nodes):
        if partition[i] == 1:
            for j in range(0,number_nodes):
                if group[i] == group[j]:
                    if(j != i) and (partition[j] == 1) and(adjM[i][j]!=0):
                        hit += 1
                    if(j !=i) and (partition[j] == 4 or partition[j] == 3) and(adjM[i][j]!=0):
                        hit -= 0.5
                    if(j != i) and (partition[j] == 2) and(adjM[i][j]!=0):
                       hit -= 1
                elif adjM[i][j] != 0:
                    hit -= 0.2
        elif partition[i] == 2:
            for j in range(0,number_nodes):
                if group[i] == group[j]:
                    if(j != i) and (partition[j] !=4 ) and(adjM[i][j]!=0):
                        hit += 1
                    if(j !=i) and (partition[j] == 4) and(adjM[i][j]!=0):
                        hit -= 0.5
                elif adjM[i][j]!=0:
                    hit -= 0.2
        elif partition[i] == 3:
            for j in range(0,number_nodes):
                if group[i] == group[j]:
                    if (j != i ) and (partition[j] == 1 or partition[j] == 2) and (adjM[i][j]!=0):
                        hit -= 0.5
                    elif (j != i ) and (partition[j] == 3 or partition[j] == 4) and (adjM[i][j]!=0):
                        hit -= 0.5
                elif adjM[i][j] != 0:
                    hit -= 0.2
        elif partition[i] == 4:
            for j in range(0,number_nodes):
                if group[i] == group[j]:
                    if(j != i) and (partition[j] == 1) and(adjM[i][j]!=0):
                        hit += 1
                    if (j != i ) and (partition[j] ==2) and (adjM[i][j]!=0):
                        hit -= 0.5
                    if (j != i ) and (partition[j] == 3 or partition[j] == 4) and (adjM[i][j]!=0):
                        hit -= 0.5
                elif adjM[i][j] != 0:
                    hit -= 0.2

    max_hit = num_Cin*(num_Cin - 1) + num_Cout *(num_Cout - 1 + num_Cin +num_Pin) + num_Pout*num_Cin
    Q = hit/max_hit
    
    return Q

def calculate_Q_M(adjM,partition,group,group_members):
    hit = 0
    num_Cin = np.count_nonzero(partition == 1 )
    num_Cout = np.count_nonzero(partition == 2 )
    num_Pin = np.count_nonzero(partition == 3 )
    num_Pout = np.count_nonzero(partition == 4)
    number_nodes = len(partition)

    
    #partition(1 = Cin 2=Cout 3=Pin 4=Pout)
    for i in range(0,number_nodes):
        if partition[i] == 1:
            for j in range(0,number_nodes):
                if(j != i) and (partition[j] == 1) and(adjM[i][j]!=0):
                    hit += 1
                if(j !=i) and (partition[j] !=1) and(adjM[i][j]!=0):
                    hit -= 1
        elif partition[i] == 2:
            for j in range(0,number_nodes):
                if(j != i) and (partition[j] !=4 ) and(adjM[i][j]!=0):
                    hit += 1
                if(j !=i) and (partition[j] == 4) and(adjM[i][j]!=0):
                    hit -= 1
        elif partition[i] == 3:
            for j in range(0,number_nodes):
                if (j != i ) and (adjM[i][j]!=0):
                    hit -= 1
        elif partition[i] == 4:
            for j in range(0,number_nodes):
                if(j != i) and (partition[j] == 1) and(adjM[i][j]!=0):
                    hit += 1
                if (j != i ) and (partition[j] !=1) and (adjM[i][j]!=0):
                    hit -=1

    max_hit = num_Cin*(num_Cin - 1) + num_Cout *(num_Cout - 1 + num_Cin +num_Pin) + num_Pout*num_Cin
    Q = hit/max_hit
    
    return Q

def calculate_Q_SG(adjM,partition,group_members):
    if len(group_members) == 0:
        return 0,0

    hit = 0
    group_partition = np.take(partition,group_members)
    num_Cin = np.count_nonzero(group_partition == 1 )
    num_Cout = np.count_nonzero(group_partition == 2 )
    num_Pin = np.count_nonzero(group_partition == 3 )
    num_Pout = np.count_nonzero(group_partition == 4)
    for i in group_members:
        for j in group_members:
            if i!=j  and (adjM[i][j]!=0):
                if partition[i] == 1: 
                    if(partition[j] == 1):
                        hit += 1
                    if(partition[j] !=1):
                        hit -= 0.5
                elif partition[i] == 2:
                    if(partition[j] !=4 ):
                        hit += 1
                    if(partition[j] == 4):
                        hit -= 0.5
                elif partition[i] == 3:
                        hit -= 1
                elif partition[i] == 4:
                    if (partition[j] == 1):
                        hit += 1
                    if (partition[j] !=1):
                        hit -= 0.5

    max_hit = num_Cin*(num_Cin - 1) + num_Cout *(num_Cout - 1 + num_Cin +num_Pin) + num_Pout*num_Cin

    return hit,max_hit

@njit
def calculate_Q_M333(adjM,partition_old,group_old,target,par_new,group_new,remain,owns_key,owns_val,remove):

    partition = np.copy(partition_old)
    group = np.copy(group_old)
    partition[target],group[target] = par_new,group_new
    hit = 0
    num_Cin = np.count_nonzero(partition == 1 )
    num_Cout = np.count_nonzero(partition == 2 )
    num_Pin = np.count_nonzero(partition == 3 )
    num_Pout = np.count_nonzero(partition == 4)
    number_nodes = len(partition)

    #partition(1 = Cin 2=Cout 3=Pin 4=Pout)
    for i in remain:
        if partition[i] == 1:
            for j in remain:
                if group[i] == group[j]:
                    if(j != i) and (partition[j] == 1) and(adjM[i][j]!=0):
                        hit += 1
                    if(j !=i) and (partition[j] == 4 or partition[j] == 3) and(adjM[i][j]!=0):
                        hit -= 0.5
                    if(j != i) and (partition[j] == 2) and(adjM[i][j]!=0):
                       hit -= 1
                elif adjM[i][j] != 0:
                    hit -= 0.2
            for j in remove:
                if group[i] == group[j]:
                    if adjM[i][j] != 0:
                        hit -= 0.5

                    elif adjM[j][i] !=0:
                        if partition[j] == 4:
                            hit += 1
                        elif partition[j] == 3:
                            hit -= 0.5 
            # if i in owns:
            #     for j in owns[i]:
            #         if j != i:
            #             if adjM[i][j] != 0:
            #                 hit -= 0.5
            #             if adjM[j][i] != 0:
            #                 if partition[j] == 4:
            #                     hit += 1
            #                 elif partition[j] == 3:
            #                     hit -= 0.5
            #                 else:
            #                     raise ValueError("trivial partition error, node partition:" + str(j) +" " +str(partition[j]))




        elif partition[i] == 2:
            for j in remain:
                if group[i] == group[j]:
                    if(j != i) and (partition[j] !=4 ) and(adjM[i][j]!=0):
                        hit += 1
                    if(j !=i) and (partition[j] == 4) and(adjM[i][j]!=0):
                        hit -= 0.5
                elif adjM[i][j]!=0:
                    hit -= 0.2

            for j in remove:
                if group[i] == group[j]:
                    if adjM[j][i] != 0:
                        hit -= 0.5
                        
                    elif adjM[j][i] !=0:
                        if partition[j] == 3:
                            hit += 1
                        elif partition[j] == 4:
                            hit -= 0.5 
                    
        elif partition[i] == 3:
            for j in remain:
                if group[i] == group[j]:
                    if (j != i ) and (partition[j] == 1 or partition[j] == 2) and (adjM[i][j]!=0):
                        hit -= 0.5
                    elif (j != i ) and (partition[j] == 3 or partition[j] == 4) and (adjM[i][j]!=0):
                        hit -= 0.5
                elif adjM[i][j] != 0:
                    hit -= 0.2
            key_idx = np.where(owns_key == i)[0]
            if len(key_idx) != 0 :
                value = owns_val[key_idx[0]]
                hit -= 0.5 * value


                
        elif partition[i] == 4:
            for j in remain:
                if group[i] == group[j]:
                    if(j != i) and (partition[j] == 1) and(adjM[i][j]!=0):
                        hit += 1
                    if (j != i ) and (partition[j] ==2) and (adjM[i][j]!=0):
                        hit -= 0.5
                    if (j != i ) and (partition[j] == 3 or partition[j] == 4) and (adjM[i][j]!=0):
                        hit -= 0.5
                elif adjM[i][j] != 0:
                    hit -= 0.2
            key_idx = np.where(owns_key == i)[0]
            if len(key_idx) != 0 :
                value = owns_val[key_idx[0]]
                hit -= 0.5 * value


    max_hit = num_Cin*(num_Cin - 1) + num_Cout *(num_Cout - 1 + num_Cin +num_Pin) + num_Pout*num_Cin
    Q = hit/max_hit
    
    return Q
@njit
def calculate_Q_M3(adjS,partition_old,group_old,target,par_new,group_new,remain,owns_key,owns_val,remove):

    partition = np.copy(partition_old)
    group = np.copy(group_old)
    partition[target],group[target] = par_new,group_new
    hit = 0
    num_Cin = np.count_nonzero(partition == 1 )
    num_Cout = np.count_nonzero(partition == 2 )
    num_Pin = np.count_nonzero(partition == 3 )
    num_Pout = np.count_nonzero(partition == 4)
    number_nodes = len(partition)




    
    #partition(1 = Cin 2=Cout 3=Pin 4=Pout)
    for i in remain:
        if partition[i] == 1:
            for j in remain:
                if (j!=i) and len(np.where(adjS == i*number_nodes + j)[0]):
                    if group[i] == group[j]:
                        # a trivial node will only be in partition 3 and 4    
                        if partition[j] == 1: 
                            hit += 1
                        elif partition[j] == 2:
                            hit -=1
                        elif partition[j] == 3 or partition[j] == 4:
                            hit -= 0.5
                    else: #in this part, j wont be a trivial while not belong to i group
                        hit -= 0.2    
            for j in remove:
                if (j != i):
                    if len(np.where(adjS == i*number_nodes + j)[0]):
                        hit -= 0.5
                    if len(np.where(adjS == j*number_nodes + i)[0]):
                        if partition[j] == 4:
                            hit += 1
                        elif partition[j] == 3:
                            hit -= 0.5

        elif partition[i] == 2:
            for j in remain:
                if (j!=i) and len(np.where(adjS == i*number_nodes + j)[0]):
                    if group[i] == group[j]:
                        # a trivial node will only be in partition 3 and 4    
                        if partition[j] == 4: 
                            hit -= 0.5
                        else:
                            hit += 1
                    else: #in this part, j wont be a trivial while not belong to i group
                        hit -= 0.2    
            for j in remove:
                if (j != i):
                    if len(np.where(adjS == j*number_nodes + i)[0]):
                        hit -= 0.5
                    if len(np.where(adjS == i*number_nodes + j)[0]):
                        if partition[j] == 3:
                            hit += 1
                        elif partition[j] == 4:
                            hit -= 0.5
                    
        elif partition[i] == 3:
            for j in remain:
                if (j!=i) and len(np.where(adjS == i*number_nodes + j)[0]):
                    if group[i] == group[j]:
                        hit -= 0.5
                    else:
                        hit -= 0.2
            key_idx = np.where(owns_key == i)[0]
            if len(key_idx) != 0 :
                value = owns_val[key_idx[0]]
                hit -= 0.5 * value


                
        elif partition[i] == 4:
            for j in remain:
                if (j!=i) and len(np.where(adjS == i*number_nodes + j)[0]):
                    if group[i] == group[j]:
                        if partition[j] == 1:
                            hit += 1
                        else:
                            hit -= 0.5
                    else:
                        hit -= 0.2
            key_idx = np.where(owns_key == i)[0]
            if len(key_idx) != 0 :
                value = owns_val[key_idx[0]]
                hit -= 0.5 * value


    max_hit = num_Cin*(num_Cin - 1) + num_Cout *(num_Cout - 1 + num_Cin +num_Pin) + num_Pout*num_Cin
    Q = hit/max_hit
    
    return Q
def calculate_Q_M33(adjM,partition_old,group_old,target,par_new,group_new,remain,owns_key,owns_val,owns_list):

    partition = np.copy(partition_old)
    group = np.copy(group_old)
    partition[target],group[target] = par_new,group_new
    hit = 0
    num_Cin = np.count_nonzero(partition == 1 )
    num_Cout = np.count_nonzero(partition == 2 )
    num_Pin = np.count_nonzero(partition == 3 )
    num_Pout = np.count_nonzero(partition == 4)
    

    #partition(1 = Cin 2=Cout 3=Pin 4=Pout)
    for i in remain:
        if partition[i] == 1:
            for j in remain:
                if group[i] == group[j]:
                    if(j != i) and (partition[j] == 1) and(adjM[i][j]!=0):
                        hit += 1
                    if(j !=i) and (partition[j] == 4 or partition[j] == 3) and(adjM[i][j]!=0):
                        hit -= 0.5
                    if(j != i) and (partition[j] == 2) and(adjM[i][j]!=0):
                       hit -= 1
                elif adjM[i][j] != 0:
                    hit -= 0.2
            
            key_idx = np.where(owns_key == i)[0]
            if len(key_idx) != 0 :
                remove = owns_list[key_idx[0]]
                for j in remove:
                    if adjM[i][j] != 0:
                        hit -= 0.5

                    elif adjM[j][i] !=0:
                        if partition[j] == 4:
                            hit += 1
                        elif partition[j] == 3:
                            hit -= 0.5 
        # if i in owns:
            #     for j in owns[i]:
            #         if j != i:
            #             if adjM[i][j] != 0:
            #                 hit -= 0.5
            #             if adjM[j][i] != 0:
            #                 if partition[j] == 4:
            #                     hit += 1
            #                 elif partition[j] == 3:
            #                     hit -= 0.5
            #                 else:
            #                     raise ValueError("trivial partition error, node partition:" + str(j) +" " +str(partition[j]))




        elif partition[i] == 2:
            for j in remain:
                if group[i] == group[j]:
                    if(j != i) and (partition[j] !=4 ) and(adjM[i][j]!=0):
                        hit += 1
                    if(j !=i) and (partition[j] == 4) and(adjM[i][j]!=0):
                        hit -= 0.5
                elif adjM[i][j]!=0:
                    hit -= 0.2

            key_idx = np.where(owns_key == i)[0]
            if len(key_idx) != 0 :
                remove = owns_list[key_idx[0]]
                for j in remove:
                        if adjM[j][i] != 0:
                            hit -= 0.5
                            
                        elif adjM[j][i] !=0:
                            if partition[j] == 3:
                                hit += 1
                            elif partition[j] == 4:
                                hit -= 0.5 
                    
        elif partition[i] == 3:
            for j in remain:
                if group[i] == group[j]:
                    if (j != i ) and (partition[j] == 1 or partition[j] == 2) and (adjM[i][j]!=0):
                        hit -= 0.5
                    elif (j != i ) and (partition[j] == 3 or partition[j] == 4) and (adjM[i][j]!=0):
                        hit -= 0.5
                elif adjM[i][j] != 0:
                    hit -= 0.2
            key_idx = np.where(owns_key == i)[0]
            if len(key_idx) != 0 :
                value = owns_val[key_idx[0]]
                hit -= 0.5 * value


                
        elif partition[i] == 4:
            for j in remain:
                if group[i] == group[j]:
                    if(j != i) and (partition[j] == 1) and(adjM[i][j]!=0):
                        hit += 1
                    if (j != i ) and (partition[j] ==2) and (adjM[i][j]!=0):
                        hit -= 0.5
                    if (j != i ) and (partition[j] == 3 or partition[j] == 4) and (adjM[i][j]!=0):
                        hit -= 0.5
                elif adjM[i][j] != 0:
                    hit -= 0.2
            key_idx = np.where(owns_key == i)[0]
            if len(key_idx) != 0 :
                value = owns_val[key_idx[0]]
                hit -= 0.5 * value


    max_hit = num_Cin*(num_Cin - 1) + num_Cout *(num_Cout - 1 + num_Cin +num_Pin) + num_Pout*num_Cin
    Q = hit/max_hit
    
    return Q

@njit
def calculate_Q_reverse(adjS,partition_old,group_old,target,par_new,group_new):
    partition = np.copy(partition_old)
    group = np.copy(group_old)
    partition[target],group[target] = par_new,group_new
    hit = 0
    num_Cin = np.count_nonzero(partition == 1 )
    num_Cout = np.count_nonzero(partition == 2 )
    num_Pin = np.count_nonzero(partition == 3 )
    num_Pout = np.count_nonzero(partition == 4)
    number_nodes = len(partition)


    for sidx in adjS:
        i, j = divmod(sidx,number_nodes)
        if group[i] == group[j]:
            if partition[i] == 1:
                if partition[j] == 3 or partition[j] == 4:
                    hit -= 0.5
                elif partition[j] == 1: hit += 1
                elif partition[j] == 2: hit -= 0.5
            elif partition[i] == 2:
                if partition[j] == 2: hit -= 0.5
                else: hit += 1
            elif partition[i] == 3:
                hit -= 0.5
            elif partition[i] == 4:
                if partition[j] == 1: hit += 1
                else: hit -= 0.5

        else:
            hit -= 0.2



    max_hit = num_Cin*(num_Cin - 1) + num_Cout *(num_Cout - 1 + num_Cin +num_Pin) + num_Pout*num_Cin
    if max_hit == 0:
        Q = - math.inf
        return Q
    else:
        Q = hit/max_hit
    
    return Q
def find_CP(adjM,G):
    number_of_nodes = G.number_of_nodes()

    #partition(1 = Cin 2=Cout 3=Pin 4=Pout)
    partition = np.random.randint(1, 5,number_of_nodes)
    order = np.arange(number_of_nodes)

    Q = calculate_Q(adjM,partition)
    while(True):
        np.random.shuffle(order)
        incre = 0
        for i in order:
            par_old = partition[i]
            Q_1 = Q_2 = Q_3 = Q_4 = 0
            if par_old !=1:
                par_1 = np.copy(partition)
                par_1[i] = 1
                Q_1 = calculate_Q(adjM,par_1)
            if par_old !=2:
                par_2 = np.copy(partition)
                par_2[i] = 2
                Q_2 = calculate_Q(adjM,par_2)
            if par_old !=3:
                par_3 = np.copy(partition)
                par_3[i] = 3
                Q_3 = calculate_Q(adjM,par_3)
            if par_old !=4:
                par_4 = np.copy(partition)
                par_4[i] = 4
                Q_4 = calculate_Q(adjM,par_4)

            new_Q = Q
            par_new = par_old
            diff = 0
            if Q_1> new_Q:
                par_new = 1
                diff = Q_1 - Q
                new_Q = Q_1
            if Q_2 > new_Q:
                par_new = 2
                diff = Q_2 - Q
                new_Q = Q_2
            if Q_3 > new_Q:
                par_new = 3
                diff = Q_3 - Q
                new_Q = Q_3
            if Q_4 > new_Q:
                par_new = 4
                diff = Q_4 - Q
                new_Q = Q_4

            incre += diff
            partition[i] = par_new
            Q = new_Q

        if (incre <= 10**-8):
            break
    
    return partition


def find_CP_M(adjM,G):
    number_of_nodes = G.number_of_nodes()

    #partition(1 = Cin 2=Cout 3=Pin 4=Pout)
    partition = np.random.randint(1, 5,number_of_nodes,dtype='int8')
    # group = np.arange(start=0, stop=number_of_nodes)
    group = np.random.randint(0,number_of_nodes,number_of_nodes)
    order = np.arange(number_of_nodes)
    group_members = [ [] for _ in range(np.max(group)+1)]


    for n in G.nodes:
        group_members[group[n]].append(n)
    hits = np.zeros(len(group_members),dtype = 'int32')
    maxs = np.zeros(len(group_members),dtype = 'int32')

    for i,groupi in enumerate(group_members):
        hits[i],maxs[i] = calculate_Q_SG(adjM,partition,groupi) 

    def subQ(hit,max):
    # return the Q that substitue new hit & max after change(s)
        return (sum(hits)-hits[group[i]]+hit)/(sum(maxs)-maxs[group[i]]+max)

    Q = sum(hits)/sum(maxs)
    iters = 0
    while(True):
        iters += 1
        np.random.shuffle(order)
        incre = 0
        for i in order:
            new_Q = Q
            diff = 0
            hitspar,maxpar = 0,0
            if len(group_members[group[i]]) > 1:
                par_old = partition[i]
                par_new = par_old

                if par_old !=1:
                    par_1 = np.copy(partition)
                    par_1[i] = 1
                    hitQ1,maxQ1 = calculate_Q_SG(adjM,par_1,group_members[group[i]])
                    par_Q = subQ(hitQ1,maxQ1)
                    if par_Q> new_Q:
                        par_new = 1
                        new_Q = par_Q
                        hitspar,maxpar = hitQ1,maxQ1
                    # if maxQ1 == 0 :
                    #     Q_1 == -float('inf')
                    # else:
                    #     Q_1 = hitQ1/maxQ1
                if par_old !=2:
                    par_2 = np.copy(partition)
                    par_2[i] = 2
                    hitQ2,maxQ2 = calculate_Q_SG(adjM,par_2,group_members[group[i]])
                    par_Q = subQ(hitQ2,maxQ2)
                    if par_Q > new_Q:
                        par_new = 2
                        new_Q = par_Q
                        hitspar,maxpar = hitQ2,maxQ2
                    # if maxQ2 == 0 :
                    #     Q_2 == -float('inf')
                    # else:
                    #     Q_2 = hitQ2/maxQ2
                if par_old !=3:
                    par_3 = np.copy(partition)
                    par_3[i] = 3
                    hitQ3,maxQ3 = calculate_Q_SG(adjM,par_3,group_members[group[i]])
                    par_Q = subQ(hitQ3,maxQ3)
                    if par_Q > new_Q:
                        par_new = 3
                        new_Q = par_Q
                        hitspar,maxpar = hitQ3,maxQ3
                    # if maxQ3 == 0 :
                    #     Q_3 == -float('inf')
                    # else:
                    #     Q_3 = hitQ3/maxQ3
                if par_old !=4:
                    par_4 = np.copy(partition)
                    par_4[i] = 4
                    hitQ4,maxQ4 = calculate_Q_SG(adjM,par_4,group_members[group[i]])
                    par_Q = subQ(hitQ4,maxQ4)
                    if par_Q > new_Q:
                        par_new = 4
                        new_Q = par_Q
                        hitspar,maxpar = hitQ4,maxQ4
                    # if maxQ4 == 0 :
                    #     Q_4 == -float('inf')
                    # else:
                    #     Q_4 = hitQ4/maxQ4
                

            nbs = [e for e in nx.all_neighbors(G,i)]
            to_groups = set(np.take(group,nbs))

            base_Q = Q
            switch_to = -1
            switch_from = group[i]
            hits_from,max_from = 0,0
            hits_to,max_to = 0,0

            for to in to_groups:
                members_from = np.delete(group_members[switch_from ],np.where(group_members[switch_from ] == i))
                members_to = np.append(group_members[to],i)
                hits1,max1 = calculate_Q_SG(adjM,partition,members_from)
                hits2,max2 = calculate_Q_SG(adjM,partition,members_to)

                if (np.max(maxs) == 0):
                    if(max1+max2 != 0):
                        temp_Q = (hits1+hits2)/(max1+max2)
                    else:
                        temp_Q = 0
                else:
                    temp_Q = (np.sum(hits) + hits1 + hits2 - hits[switch_from ] - hits[to])/(np.sum(maxs) + max1 +max2 -maxs[switch_from] - maxs[to])

                if temp_Q >= base_Q:
                    base_Q = temp_Q
                    switch_to = to
                    hits_from,max_from = hits1,max1
                    hits_to, max_to = hits2,max2      


            if base_Q > new_Q and switch_to != -1:
                # the result of group switching is better
                diff = base_Q - Q
                Q = base_Q
                group_members[switch_from] = np.delete(group_members[switch_from],np.where(group_members[switch_from] == i))
                group[i] = switch_to
                group_members[switch_to] = np.append(group_members[switch_to],i)

                # update hit/max
                hits[switch_from],maxs[switch_from] = hits_from,max_from
                hits[switch_to],maxs[switch_to] = hits_to,max_to

            elif new_Q > base_Q and par_new != par_old:
                # the result of partition switching is better
                diff = new_Q - Q
                Q = new_Q
                partition[i] = par_new
                # update hit/max
                hits[group[i]],maxs[group[i]] = hitspar,maxpar

            incre += diff

        if (incre <= 10**-8 and iters >= 10):
            print("iters:"+str(iters))
            print("incre:"+str(incre))
            break
    
    return partition,group

def find_CP_M2(adjM,G):
    

    number_of_nodes = G.number_of_nodes()
    remain = []
    remove = []
    for node,degree in dict(G.degree()).items():
        if degree >=2:
            remain.append(node)
        elif degree<2:
            remove.append(node)

    remain = np.array(remain)
    remove = np.array(remove)
    belongs = {}
    for node in remove: 
        for neib in nx.all_neighbors(G,node):
            belongs[node] = neib

    owns = {}
    owning = {}
    for node,belon in belongs.items():
        if belon in owns:
            owns[belon].append(node)
            owning[belon] += 1
        else:
            owns[belon] = [node]
            owning[belon] = 1

    owns_key = np.fromiter(owns.keys(),dtype='int32')
    owns_val = np.fromiter(owning.values(),dtype = 'int32')

    owns_list = [np.array(owns[i]) for i in owns]

    max_len = 0 
    for li in owns_list:
        if len(li) > max_len:
            max_len = len(li)
    
    for li in owns_list:
        if len(li) < max_len:
            zero = np.zeros(max_len - len(li))
            li = np.concatenate((li,zero))
    
    owns_list = np.array(owns_list)


    trivial_partition = {}
    signi_partition = {}

    for node in remove:
        if G.in_degree(node) == 1:
            trivial_partition[node] = 3
        elif G.out_degree(node) == 1:
            trivial_partition[node] = 4
        else:
            raise ValueError("removed node degree is not 1")
    
    for node in remain:
        ins = G.in_degree(node)
        outs = G.out_degree(node)

        if ins > outs:
            signi_partition[node] = 1
        elif ins < outs:
            signi_partition[node] = 2
        else:
            signi_partition[node] = np.random.randint(1,5)


    #partition(1 = Cin 2=Cout 3=Pin 4=Pout)
    partition = np.zeros(number_of_nodes)
    for idx in range(number_of_nodes):
        if idx in trivial_partition:
            partition[idx] = trivial_partition[idx]
        elif idx in signi_partition:
            partition[idx] = signi_partition[idx]
        else:
            raise ValueError("idx does not exist")

    # group = np.arange(start=0, stop=number_of_nodes)
    group = np.arange(number_of_nodes)

    for owner,owned in owns.items():
        group[owned] = owner

    order = np.array(remain)

    #group_members = [ [] for _ in range(np.max(group)+1)]


    #for n in G.nodes:
    #    group_members[group[n]].append(n)

    print("run started...")
    Q = - math.inf
    iters = 0
    last_incre = 1
    while(True):
        iters += 1
        np.random.shuffle(order)
        incre = 0
        for i in order:
            new_Q = Q
            diff = 0

            #if len(group_members[group[i]]) > 1:
            par_old = partition[i]
            par_new = par_old

            if par_old !=1:
                par_Q = calculate_Q_M3(adjM,partition,group,i,1,group[i],remain,owns_key,owns_val,remove)
                if par_Q> new_Q:
                    par_new = 1
                    new_Q = par_Q

                # if maxQ1 == 0 :
                #     Q_1 == -float('inf')
                # else:
                #     Q_1 = hitQ1/maxQ1
            if par_old !=2:
                par_Q = calculate_Q_M3(adjM,partition,group,i,2,group[i],remain,owns_key,owns_val,remove)
                if par_Q > new_Q:
                    par_new = 2
                    new_Q = par_Q
                # if maxQ2 == 0 :
                #     Q_2 == -float('inf')
                # else:
                #     Q_2 = hitQ2/maxQ2
            if par_old !=3:
                par_Q = calculate_Q_M3(adjM,partition,group,i,3,group[i],remain,owns_key,owns_val,remove)
                if par_Q > new_Q:
                    par_new = 3
                    new_Q = par_Q

                # if maxQ3 == 0 :
                #     Q_3 == -float('inf')
                # else:
                #     Q_3 = hitQ3/maxQ3
            if par_old !=4:
                par_Q = calculate_Q_M3(adjM,partition,group,i,4,group[i],remain,owns_key,owns_val,remove)
                if par_Q > new_Q:
                    par_new = 4
                    new_Q = par_Q
                # if maxQ4 == 0 :
                #     Q_4 == -float('inf')
                # else:
                #     Q_4 = hitQ4/maxQ4
                

            nbs = [e for e in nx.all_neighbors(G,i)]
            to_groups = set(np.take(group,nbs))

            base_Q = Q
            switch_to = -1
            switch_from = group[i]


            def wrapperfunc(target_group):
                return target_group, calculate_Q_M3(adjM,partition,group,i,partition[i],target_group,remain,owns_key,owns_val,remove)

            processes = []
            with ThreadPoolExecutor() as ex:
                processes.append(ex.map(wrapperfunc,to_groups))

            
            for results in processes[0]:
                temp_Q = results[1]
                to = results[0]

                if temp_Q >= base_Q:
                    base_Q = temp_Q
                    switch_to = to
                    
            # for to in to_groups:
            #     temp_Q = calculate_Q_M2(adjM,partition,group,i,partition[i],to)
                
            #     if temp_Q >= base_Q:
            #         base_Q = temp_Q
            #         switch_to = to

            if base_Q > new_Q and switch_to != -1:
                # the result of group switching is better
                diff = base_Q - Q
                Q = base_Q
                #group_members[switch_from] = np.delete(group_members[switch_from],np.where(group_members[switch_from] == i))
                group[i] = switch_to
                if i in owns:
                    group[owns[i]] = switch_to
                #group_members[switch_to] = np.append(group_members[switch_to],i)
            elif new_Q > base_Q and par_new != par_old:
                # the result of partition switching is better
                diff = new_Q - Q
                Q = new_Q
                partition[i] = par_new
                # update hit/max

            incre += diff

        print("still running: " + str(iters))
        
        if (incre <= 10**-10 and last_incre <= 10**-10  and iters >= 4):
            print("iters:"+str(iters))
            print("incre:"+str(incre))
            break
        last_incre = incre    

    return partition,group

def find_CP_M3(adjS,G):
    

    number_of_nodes = G.number_of_nodes()
    remain = []
    remove = []
    for node,degree in dict(G.degree()).items():
        if degree >=2:
            remain.append(node)
        elif degree<2:
            remove.append(node)

    remain = np.array(remain)
    remove = np.array(remove)
    belongs = {}
    for node in remove: 
        for neib in nx.all_neighbors(G,node):
            belongs[node] = neib

    owns = {}
    owning = {}
    for node,belon in belongs.items():
        if belon in owns:
            owns[belon].append(node)
            owning[belon] += 1
        else:
            owns[belon] = [node]
            owning[belon] = 1

    owns_key = np.fromiter(owns.keys(),dtype='int32')
    owns_val = np.fromiter(owning.values(),dtype = 'int32')

    owns_list = [np.array(owns[i]) for i in owns]

    max_len = 0 
    for li in owns_list:
        if len(li) > max_len:
            max_len = len(li)
    
    for li in owns_list:
        if len(li) < max_len:
            zero = np.zeros(max_len - len(li))
            li = np.concatenate((li,zero))
    
    owns_list = np.array(owns_list)


    trivial_partition = {}
    signi_partition = {}

    for node in remove:
        if G.in_degree(node) == 1:
            trivial_partition[node] = 3
        elif G.out_degree(node) == 1:
            trivial_partition[node] = 4
        else:
            raise ValueError("removed node degree is not 1")
    
    for node in remain:
        ins = G.in_degree(node)
        outs = G.out_degree(node)

        if ins > outs:
            signi_partition[node] = 1
        elif ins < outs:
            signi_partition[node] = 2
        else:
            signi_partition[node] = np.random.randint(1,5)


    #partition(1 = Cin 2=Cout 3=Pin 4=Pout)
    partition = np.zeros(number_of_nodes)
    for idx in range(number_of_nodes):
        if idx in trivial_partition:
            partition[idx] = trivial_partition[idx]
        elif idx in signi_partition:
            partition[idx] = signi_partition[idx]
        else:
            raise ValueError("idx does not exist")

    # group = np.arange(start=0, stop=number_of_nodes)
    group = np.arange(number_of_nodes)

    for owner,owned in owns.items():
        group[owned] = owner

    order = np.array(remain)

    #group_members = [ [] for _ in range(np.max(group)+1)]


    #for n in G.nodes:
    #    group_members[group[n]].append(n)

    print("run started...")
    Q = - math.inf
    iters = 0
    last_incre = 1
    while(True):
        iters += 1
        np.random.shuffle(order)
        incre = 0
        for i in order:
            new_Q = Q
            diff = 0

            #if len(group_members[group[i]]) > 1:
            par_old = partition[i]
            par_new = par_old

            if par_old !=1:
                par_Q = calculate_Q_M3(adjS,partition,group,i,1,group[i],remain,owns_key,owns_val,remove)
                if par_Q> new_Q:
                    par_new = 1
                    new_Q = par_Q

                # if maxQ1 == 0 :
                #     Q_1 == -float('inf')
                # else:
                #     Q_1 = hitQ1/maxQ1
            if par_old !=2:
                par_Q = calculate_Q_M3(adjS,partition,group,i,2,group[i],remain,owns_key,owns_val,remove)
                if par_Q > new_Q:
                    par_new = 2
                    new_Q = par_Q
                # if maxQ2 == 0 :
                #     Q_2 == -float('inf')
                # else:
                #     Q_2 = hitQ2/maxQ2
            if par_old !=3:
                par_Q = calculate_Q_M3(adjS,partition,group,i,3,group[i],remain,owns_key,owns_val,remove)
                if par_Q > new_Q:
                    par_new = 3
                    new_Q = par_Q

                # if maxQ3 == 0 :
                #     Q_3 == -float('inf')
                # else:
                #     Q_3 = hitQ3/maxQ3
            if par_old !=4:
                par_Q = calculate_Q_M3(adjS,partition,group,i,4,group[i],remain,owns_key,owns_val,remove)
                if par_Q > new_Q:
                    par_new = 4
                    new_Q = par_Q
                # if maxQ4 == 0 :
                #     Q_4 == -float('inf')
                # else:
                #     Q_4 = hitQ4/maxQ4
                

            nbs = [e for e in nx.all_neighbors(G,i)]
            to_groups = set(np.take(group,nbs))

            base_Q = Q
            switch_to = -1
            switch_from = group[i]


            def wrapperfunc(target_group):
                return target_group, calculate_Q_M3(adjS,partition,group,i,partition[i],target_group,remain,owns_key,owns_val,remove)

            processes = []
            with ThreadPoolExecutor() as ex:
                processes.append(ex.map(wrapperfunc,to_groups))

            
            for results in processes[0]:
                temp_Q = results[1]
                to = results[0]

                if temp_Q >= base_Q:
                    base_Q = temp_Q
                    switch_to = to
                    
            # for to in to_groups:
            #     temp_Q = calculate_Q_M2(adjM,partition,group,i,partition[i],to)
                
            #     if temp_Q >= base_Q:
            #         base_Q = temp_Q
            #         switch_to = to

            if base_Q > new_Q and switch_to != -1:
                # the result of group switching is better
                diff = base_Q - Q
                Q = base_Q
                #group_members[switch_from] = np.delete(group_members[switch_from],np.where(group_members[switch_from] == i))
                group[i] = switch_to
                if i in owns:
                    group[owns[i]] = switch_to
                #group_members[switch_to] = np.append(group_members[switch_to],i)
            elif new_Q > base_Q and par_new != par_old:
                # the result of partition switching is better
                diff = new_Q - Q
                Q = new_Q
                partition[i] = par_new
                # update hit/max

            incre += diff

        print("still running: " + str(iters))
        
        if (incre <= 10**-10 and last_incre <= 10**-10  and iters >= 4):
            print("iters:"+str(iters))
            print("incre:"+str(incre))
            break
        last_incre = incre    

    return partition,group
def find_CP_M4(adjS,G):
    

    number_of_nodes = G.number_of_nodes()
    remain = []
    remove = []
    for node,degree in dict(G.degree()).items():
        if degree >=2:
            remain.append(node)
        elif degree<2:
            remove.append(node)

    remain = np.array(remain)
    remove = np.array(remove)
    belongs = {}
    for node in remove: 
        for neib in nx.all_neighbors(G,node):
            belongs[node] = neib

    owns = {}
    owning = {}
    for node,belon in belongs.items():
        if belon in owns:
            owns[belon].append(node)
            owning[belon] += 1
        else:
            owns[belon] = [node]
            owning[belon] = 1

    # owns_key = np.fromiter(owns.keys(),dtype='int32')
    # owns_val = np.fromiter(owning.values(),dtype = 'int32')

    owns_list = [np.array(owns[i]) for i in owns]

    max_len = 0 
    for li in owns_list:
        if len(li) > max_len:
            max_len = len(li)
    
    for li in owns_list:
        if len(li) < max_len:
            zero = np.zeros(max_len - len(li))
            li = np.concatenate((li,zero))
    
    owns_list = np.array(owns_list)


    trivial_partition = {}
    signi_partition = {}

    for node in remove:
        if G.in_degree(node) == 1:
            trivial_partition[node] = 3
        elif G.out_degree(node) == 1:
            trivial_partition[node] = 4
        elif nx.all_neighbors(G,node) == 0:
            pass
        else:
            raise ValueError("removed node degree is not 1")
    
    for node in remain:
        ins = G.in_degree(node)
        outs = G.out_degree(node)

        if ins > outs:
            signi_partition[node] = 1
        elif ins < outs:
            signi_partition[node] = 2
        else:
            signi_partition[node] = np.random.randint(1,5)


    #partition(1 = Cin 2=Cout 3=Pin 4=Pout)
    partition = np.zeros(number_of_nodes)
    for idx in range(number_of_nodes):
        if idx in trivial_partition:
            partition[idx] = trivial_partition[idx]
        elif idx in signi_partition:
            partition[idx] = signi_partition[idx]
        else:
            raise ValueError("idx does not exist")

    # group = np.arange(start=0, stop=number_of_nodes)
    group = np.arange(number_of_nodes)

    for owner,owned in owns.items():
        group[owned] = owner

    order = np.array(remain)

    #group_members = [ [] for _ in range(np.max(group)+1)]


    #for n in G.nodes:
    #    group_members[group[n]].append(n)

    print("run started...")
    Q = - math.inf
    iters = 0
    last_incre = 1
    while(True):
        iters += 1
        np.random.shuffle(order)
        incre = 0
        for i in order:
            new_Q = Q
            diff = 0

            #if len(group_members[group[i]]) > 1:
            par_old = partition[i]
            par_new = par_old

            if par_old !=1:
                par_Q = calculate_Q_reverse(adjS,partition,group,i,1,group[i])
                if par_Q> new_Q:
                    par_new = 1
                    new_Q = par_Q

                # if maxQ1 == 0 :
                #     Q_1 == -float('inf')
                # else:
                #     Q_1 = hitQ1/maxQ1
            if par_old !=2:
                par_Q = calculate_Q_reverse(adjS,partition,group,i,2,group[i])
                if par_Q > new_Q:
                    par_new = 2
                    new_Q = par_Q
                # if maxQ2 == 0 :
                #     Q_2 == -float('inf')
                # else:
                #     Q_2 = hitQ2/maxQ2
            if par_old !=3:
                par_Q = calculate_Q_reverse(adjS,partition,group,i,3,group[i])
                if par_Q > new_Q:
                    par_new = 3
                    new_Q = par_Q

                # if maxQ3 == 0 :
                #     Q_3 == -float('inf')
                # else:
                #     Q_3 = hitQ3/maxQ3
            if par_old !=4:
                par_Q = calculate_Q_reverse(adjS,partition,group,i,4,group[i])
                if par_Q > new_Q:
                    par_new = 4
                    new_Q = par_Q
                # if maxQ4 == 0 :
                #     Q_4 == -float('inf')
                # else:
                #     Q_4 = hitQ4/maxQ4
                

            nbs = [e for e in nx.all_neighbors(G,i)]
            to_groups = set(np.take(group,nbs))

            base_Q = Q
            switch_to = -1
            switch_from = group[i]


            def wrapperfunc(target_group):
                return target_group, calculate_Q_reverse(adjS,partition,group,i,partition[i],target_group)

            processes = []
            with ThreadPoolExecutor() as ex:
                processes.append(ex.map(wrapperfunc,to_groups))

            
            for results in processes[0]:
                temp_Q = results[1]
                to = results[0]

                if temp_Q >= base_Q:
                    base_Q = temp_Q
                    switch_to = to
                    
            # for to in to_groups:
            #     temp_Q = calculate_Q_M2(adjM,partition,group,i,partition[i],to)
                
            #     if temp_Q >= base_Q:
            #         base_Q = temp_Q
            #         switch_to = to

            if base_Q > new_Q and switch_to != -1:
                # the result of group switching is better
                diff = base_Q - Q
                Q = base_Q
                #group_members[switch_from] = np.delete(group_members[switch_from],np.where(group_members[switch_from] == i))
                group[i] = switch_to
                if i in owns:
                    group[owns[i]] = switch_to
                #group_members[switch_to] = np.append(group_members[switch_to],i)
            elif new_Q > base_Q and par_new != par_old:
                # the result of partition switching is better
                diff = new_Q - Q
                Q = new_Q
                partition[i] = par_new
                # update hit/max

            incre += diff

        print("still running: " + str(iters))
        
        if (incre <= 10**-10 and last_incre <= 10**-10  and iters >= 4):
            print("iters:"+str(iters))
            print("incre:"+str(incre))
            break
        last_incre = incre    
    print("Final Q:" + str(Q))
    return partition,group

def main():

    # df = pd.read_csv("host_relations.csv", skiprows = 0)

    # #the starting node index is zero
    # index = 0
    # node_list = []

    # #when a new url appears, replace it with a new int number as its node index
    # for col in range(len(df.columns)):
    #     for row in range(len(df)):
    #         if(isinstance(df.iat[row,col],str)):
    #             node_list.append([index,df.iat[row,col]])
    #             df = df.mask(df == df.iat[row,col],index)
    #             index += 1

    # #construct network from edgelist
    # e_list = df.values.tolist()
    # e_list_b = df.values.tolist()
    # map_list = {}
    # index = 0
    # for pairs in e_list:
    #     if pairs[0] not in map_list:
    #         map_list[pairs[0]] = index
    #         index += 1

    #     if pairs[1] not in map_list:
    #         map_list[pairs[1]] = index
    #         index += 1

    # for pairs in e_list:
    #     pairs[0] = map_list[pairs[0]]
    #     pairs[1] = map_list[pairs[1]]

    # G = nx.DiGraph()
    # G.add_edges_from(e_list)

    df = pd.read_excel("dexten.xlsx", skiprows = 0, dtype='int64')
    e_list = df.values.tolist()
    e_list_b = df.values.tolist()
    map_list = {}
    index = 0
    for pairs in e_list:
        if pairs[0] not in map_list:
            map_list[pairs[0]] = index
            index += 1

        if pairs[1] not in map_list:
            map_list[pairs[1]] = index
            index += 1

    for pairs in e_list:
        pairs[0] = map_list[pairs[0]]
        pairs[1] = map_list[pairs[1]]

    G = nx.DiGraph()
    G.add_edges_from(e_list)
    # np.save('map_list.npy', map_list) 




    #G = nx.generators.directed.random_k_out_graph(10, 3, 0.5)

    # F = nx.scale_free_graph(200,alpha=0.2,beta=0.6,gamma=0.2,seed = 123)
    # G = nx.empty_graph(F.number_of_nodes(),create_using=nx.DiGraph())
    # ed_list = []
    # for e in F.edges():
    #     if e not in ed_list and e[0] != e[1]:
    #         ed_list.append(e)
    # G.add_edges_from(ed_list)

    # pos = nx.layout.spring_layout(G)

    # node_sizes = [3 + 10 * i for i in range(len(G))]
    # M = G.number_of_edges()
    # edge_colors = range(2, M + 2)

    # nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="blue")
    # edges = nx.draw_networkx_edges(
    #     G,
    #     pos,
    #     node_size=node_sizes,
    #     arrowstyle="->",
    #     arrowsize=10,
    #     edge_color=edge_colors,
    #     edge_cmap=plt.cm.Blues,
    #     width=2,
    # )
    # labels = nx.draw_networkx_labels(G,pos)

    # pc = mpl.collections.PatchCollection(edges, cmap=plt.cm.Blues)
    # pc.set_array(edge_colors)
    # plt.colorbar(pc)

    # ax = plt.gca()
    # ax.set_axis_off()
    # plt.show()

    #adj = nx.adjacency_matrix(G).toarray()

    # G = nx.DiGraph()
    # df = pd.read_csv("arti_app.csv", skiprows = 0)
    # sources = df["Source"]
    # targets = df["Target"]
    # edges = [None]* len(sources)
    # for i in range(len(sources)):
    #     edges[i] = (sources[i],targets[i])
    # G.add_edges_from(edges)

    
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    print(num_nodes)
    print(num_edges)
    print(num_edges/num_nodes)
    adjS = np.empty(num_edges,dtype=np.uint32)
    idx = 0
    for node in G.nodes():
        for neib in sorted(G.neighbors(node)):
            adjS[idx] = num_nodes * node + neib
            idx += 1
    part,group= find_CP_M4(adjS,G)
    group_map = {}
    group_map_re = {}
    idx = 0
    for g in group:
        if g not in group_map:
            group_map[g] = idx
            group_map_re[idx] = g
            idx += 1
    for i in range(len(group)):
        group[i] = group_map[group[i]]
    

    nx.write_gml(G,"dxtenbench.gml")
    with open('dxtenbench.csv', mode='w', newline='') as export_file:
        export_writer = csv.writer(export_file,delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        export_writer.writerow(['Id','coreness','group'])
        for node in G.nodes:
            export_writer.writerow([node,part[node],group[node]])

    print(part)
    print(group)

if __name__ == "__main__":
    main()