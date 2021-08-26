
import networkx as nx
import csv
import numpy as np
import pandas as pd
from directed import calculate_Q_reverse
from directed import find_CP_M4


def main():

    # F = nx.read_gml("FINAL200_4.gml")
    # G = nx.convert_node_labels_to_integers(F)

    df = pd.read_excel("dexten.xlsx", skiprows = 0, dtype='int64')
    
    e_list = df.values.tolist()
    e_list_b = df.values.tolist()
    map_list = {}
    map_list_re = {}
    index = 0
    for pairs in e_list:
        if pairs[0] not in map_list:
            map_list[pairs[0]] = index
            map_list_re[index] = pairs[0]
            index += 1

        if pairs[1] not in map_list:
            map_list[pairs[1]] = index
            map_list_re[index] = pairs[1]
            index += 1

    for pairs in e_list:
        pairs[0] = map_list[pairs[0]]
        pairs[1] = map_list[pairs[1]]

    G = nx.DiGraph()
    G.add_edges_from(e_list)


    df_results = pd.read_csv("FINAL200_4.csv")
    
    
    groups = {}
    for group in df_results["group"]:
        if group not in groups:
            groups[group] = 1
        else:
            groups[group] += 1

    next_layer = []
    for n in range(G.number_of_nodes()):
        if df_results['coreness'][n] == 1 or df_results['coreness'][n] == 2:
            next_layer.append(n)


    H = G.subgraph(next_layer)
    isolates = list(nx.isolates(H))
    for x in isolates: next_layer.remove(x)
    F = G.subgraph(next_layer)
    mapping = {}
    mapping_re = {}
    for n,key in enumerate(F.nodes()):
        if key not in mapping:
            mapping[key] = n
            mapping_re[n] = key

    new_groups = list(df_results["group"]) # new groups record the group partition after this layer's process finish
    old_groups = [None] * len(mapping)
    for re in mapping_re:
        old_groups[re] = new_groups[mapping_re[re]]
    #old groups record the current layer nodes' group partition in previous layer
    

    F = nx.relabel_nodes(F,mapping)
    num_nodes = F.number_of_nodes()
    num_edges = F.number_of_edges()
    adjS = np.empty(num_edges,dtype=np.uint32)
    idx = 0
    for node in F.nodes():
        for neib in sorted(F.neighbors(node)):
            adjS[idx] = num_nodes * node + neib
            idx += 1
    p, g = find_CP_M4(adjS,F)
    print(p)
    print(g)
    old_groups = np.array(old_groups)
    new_groups = np.array(new_groups)
    for i in range(len(new_groups)):
        gr = new_groups[i]
        if gr in old_groups:
            idx = np.where(old_groups == gr)[0][0]
            new_g = g[idx] + 100000 #offset new group for later distinguish
            new_groups[i] = new_g
            
    cast_back = np.copy(new_groups)
    partition_filled = [None] * len(cast_back)
    for i in range(len(cast_back)):
        if i in mapping:
            partition_filled[i] = p[mapping[i]]
        else:
            partition_filled[i] = 4.5


    for i in range(len(cast_back)):
        ng  = cast_back[i]
        if ng-100000 in mapping_re:
            cast_back[i] = mapping_re[ng-100000]

    cb_map = {}
    cb_idx = 0
    for cb in cast_back:
        if cb not in cb_map:
            cb_map[cb] = cb_idx
            cb_idx += 1

    cast_back_construct = np.copy(cast_back)
    for i in range(len(cast_back_construct)):
        cast_back_construct[i] = cb_map[cast_back_construct[i]]

    # writerow last argument:
    #   cast_back_construct[...]: the group information with a compact index
    #   cast_back[...]: the group information with index on layer-1
    with open('layer2.csv', mode='w', newline='') as export_file:
        export_writer = csv.writer(export_file,delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        export_writer.writerow(['Id','c_l2','g_l2'])
        for node in G.nodes:
            export_writer.writerow([node,partition_filled[node], cast_back_construct[node]])


#   third LAYER
    third_layer = []
    for n in next_layer:
        if partition_filled[n] == 1 or partition_filled[n] == 2:
            third_layer.append(n)


    H2 = G.subgraph(third_layer)
    isolates = list(nx.isolates(H2))
    for x in isolates: third_layer.remove(x)
    F = G.subgraph(third_layer)
    mapping = {}
    mapping_re = {}
    for n,key in enumerate(F.nodes()):
        if key not in mapping:
            mapping[key] = n
            mapping_re[n] = key

    new_groups = np.copy(cast_back) # new groups record the group partition after this layer's process finish
    old_groups = [None] * len(mapping)
    for re in mapping_re:
        old_groups[re] = new_groups[mapping_re[re]]
    #old groups record the current layer nodes' group partition in previous layer
    

    F = nx.relabel_nodes(F,mapping)
    num_nodes = F.number_of_nodes()
    num_edges = F.number_of_edges()
    adjS = np.empty(num_edges,dtype=np.uint32)
    idx = 0
    for node in F.nodes():
        for neib in sorted(F.neighbors(node)):
            adjS[idx] = num_nodes * node + neib
            idx += 1
    p, g = find_CP_M4(adjS,F)
    print(p)
    print(g)
    old_groups = np.array(old_groups)
    new_groups = np.array(new_groups)
    for i in range(len(new_groups)):
        gr = new_groups[i]
        if gr in old_groups:
            idx = np.where(old_groups == gr)[0][0]
            new_g = g[idx] + 100000 #offset new group for later distinguish
            new_groups[i] = new_g
            
    cast_back = np.copy(new_groups)
    partition_filled = [None] * len(cast_back)
    for i in range(len(cast_back)):
        if i in mapping:
            partition_filled[i] = p[mapping[i]]
        else:
            partition_filled[i] = 4.5


    for i in range(len(cast_back)):
        ng  = cast_back[i]
        if ng-100000 in mapping_re:
            cast_back[i] = mapping_re[ng-100000]

    cb_map = {}
    cb_idx = 0
    for cb in cast_back:
        if cb not in cb_map:
            cb_map[cb] = cb_idx
            cb_idx += 1

    cast_back_construct = np.copy(cast_back)
    for i in range(len(cast_back_construct)):
        cast_back_construct[i] = cb_map[cast_back_construct[i]]


    # writerow last argument:
    #   cast_back_construct[...]: the group information with a compact index
    #   cast_back[...]: the group information with index on layer-1        
    with open('layer3.csv', mode='w', newline='') as export_file:
        export_writer = csv.writer(export_file,delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        export_writer.writerow(['Id','c_l3','g_l3'])
        for node in G.nodes:
            export_writer.writerow([node,partition_filled[node],cast_back_construct[node]])

#   fourth LAYER
    fourth_layer = []
    for n in third_layer:
        if partition_filled[n] == 1 or partition_filled[n] == 2:
            fourth_layer.append(n)


    H3 = G.subgraph(fourth_layer)
    isolates = list(nx.isolates(H3))
    for x in isolates: fourth_layer.remove(x)
    F = G.subgraph(fourth_layer)
    mapping = {}
    mapping_re = {}
    for n,key in enumerate(F.nodes()):
        if key not in mapping:
            mapping[key] = n
            mapping_re[n] = key

    new_groups = np.copy(cast_back) # new groups record the group partition after this layer's process finish
    old_groups = [None] * len(mapping)
    for re in mapping_re:
        old_groups[re] = new_groups[mapping_re[re]]
    #old groups record the current layer nodes' group partition in previous layer
    

    F = nx.relabel_nodes(F,mapping)
    num_nodes = F.number_of_nodes()
    num_edges = F.number_of_edges()
    adjS = np.empty(num_edges,dtype=np.uint32)
    idx = 0
    for node in F.nodes():
        for neib in sorted(F.neighbors(node)):
            adjS[idx] = num_nodes * node + neib
            idx += 1
    p, g = find_CP_M4(adjS,F)
    print(p)
    print(g)
    old_groups = np.array(old_groups)
    new_groups = np.array(new_groups)
    for i in range(len(new_groups)):
        gr = new_groups[i]
        if gr in old_groups:
            idx = np.where(old_groups == gr)[0][0]
            new_g = g[idx] + 100000 #offset new group for later distinguish
            new_groups[i] = new_g
            
    cast_back = np.copy(new_groups)
    partition_filled = [None] * len(cast_back)
    for i in range(len(cast_back)):
        if i in mapping:
            partition_filled[i] = p[mapping[i]]
        else:
            partition_filled[i] = 4.5


    for i in range(len(cast_back)):
        ng  = cast_back[i]
        if ng-100000 in mapping_re:
            cast_back[i] = mapping_re[ng-100000]

    cb_map = {}
    cb_idx = 0
    for cb in cast_back:
        if cb not in cb_map:
            cb_map[cb] = cb_idx
            cb_idx += 1

    cast_back_construct = np.copy(cast_back)
    for i in range(len(cast_back_construct)):
        cast_back_construct[i] = cb_map[cast_back_construct[i]]

    # writerow last argument:
    #   cast_back_construct[...]: the group information with a compact index
    #   cast_back[...]: the group information with index on layer-1
    with open('layer4.csv', mode='w', newline='') as export_file:
        export_writer = csv.writer(export_file,delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        export_writer.writerow(['Id','c_l4','g_l4'])
        for node in G.nodes:
            export_writer.writerow([node,partition_filled[node],cast_back_construct[node]])


#   fifth LAYER
    fifth_layer = []
    for n in fourth_layer:
        if partition_filled[n] == 1 or partition_filled[n] == 2:
            fifth_layer.append(n)


    H3 = G.subgraph(fifth_layer)
    isolates = list(nx.isolates(H3))
    for x in isolates: fifth_layer.remove(x)
    F = G.subgraph(fifth_layer)
    mapping = {}
    mapping_re = {}
    for n,key in enumerate(F.nodes()):
        if key not in mapping:
            mapping[key] = n
            mapping_re[n] = key

    new_groups = np.copy(cast_back) # new groups record the group partition after this layer's process finish
    old_groups = [None] * len(mapping)
    for re in mapping_re:
        old_groups[re] = new_groups[mapping_re[re]]
    #old groups record the current layer nodes' group partition in previous layer
    

    F = nx.relabel_nodes(F,mapping)
    num_nodes = F.number_of_nodes()
    num_edges = F.number_of_edges()
    adjS = np.empty(num_edges,dtype=np.uint32)
    idx = 0
    for node in F.nodes():
        for neib in sorted(F.neighbors(node)):
            adjS[idx] = num_nodes * node + neib
            idx += 1
    p, g = find_CP_M4(adjS,F)
    print(p)
    print(g)
    old_groups = np.array(old_groups)
    new_groups = np.array(new_groups)
    for i in range(len(new_groups)):
        gr = new_groups[i]
        if gr in old_groups:
            idx = np.where(old_groups == gr)[0][0]
            new_g = g[idx] + 100000 #offset new group for later distinguish
            new_groups[i] = new_g
            
    cast_back = np.copy(new_groups)
    partition_filled = [None] * len(cast_back)
    for i in range(len(cast_back)):
        if i in mapping:
            partition_filled[i] = p[mapping[i]]
        else:
            partition_filled[i] = 4.5


    for i in range(len(cast_back)):
        ng  = cast_back[i]
        if ng-100000 in mapping_re:
            cast_back[i] = mapping_re[ng-100000]

    cb_map = {}
    cb_idx = 0
    for cb in cast_back:
        if cb not in cb_map:
            cb_map[cb] = cb_idx
            cb_idx += 1

    cast_back_construct = np.copy(cast_back)
    for i in range(len(cast_back_construct)):
        cast_back_construct[i] = cb_map[cast_back_construct[i]]
        
    # writerow last argument:
    #   cast_back_construct[...]: the group information with a compact index
    #   cast_back[...]: the group information with index on layer-1
    with open('layer4.csv', mode='w', newline='') as export_file:
        export_writer = csv.writer(export_file,delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        export_writer.writerow(['Id','c_l4','g_l4'])
        for node in G.nodes:
            export_writer.writerow([node,partition_filled[node],cast_back_construct[node]])

    pass

if __name__ == "__main__":
    main()