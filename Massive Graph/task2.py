import json
import time
from itertools import combinations
from pyspark import SparkConf, SparkContext
from operator import add
import random
import csv
import json
from collections import defaultdict
import os
import sys
import numpy as np

start_time = time.time()
filter_threshold = int(sys.argv[1])
input_file_path = sys.argv[2]
betweenness_output_file_path = sys.argv[3]
community_output_file_path = sys.argv[4]
conf = SparkConf().setAppName("553hw4").setMaster('local[*]')
sc = SparkContext(conf=conf)
sc.setLogLevel("WARN")
data = sc.textFile(input_file_path)
header = data.first()
process_data = data.filter(lambda i:i!=header).map(lambda i:(i.split(',')[0],i.split(',')[1]))
process_data = process_data.groupByKey().map(lambda i:(i[0],list(set(i[1])))).collect()
#print(process_data)
def find_same(process_data):
    a1,a2=set(),set()
    for i in process_data:
        for j in process_data:
            if i[0]!=j[0]:
                same = set(i[1])&set(j[1])
                len_same = len(same)
                if len_same >= filter_threshold:
                    a1.add(i[0])
                    a1.add(j[0])
                    a2.add(tuple((i[0],j[0])))
    return a1,a2

vertice,edge = find_same(process_data)
#print(edge)

def set_to_dict(element):
    default = defaultdict(set)
    for i in element:
        default[i[0]].add(i[1])
    return default

adj = set_to_dict(edge)
#print(type(adj))

def edge_calculation(my_list,older_dict,nodevalue,edgevalue,shortestpath):
    reversed_list = my_list[::-1]
    for i in reversed_list:
        older_lst = older_dict[i]
        for j in older_lst:
            shortest_value = float(shortestpath[j]/shortestpath[i])
            curr_value = nodevalue[i]*shortest_value
            edgevalue[tuple(sorted([j,i]))]+=curr_value
            nodevalue[j]+=curr_value
    return edgevalue


def breadth_first_search(root,adj):
    parent,used_set,used_lst = defaultdict(set),set(),[]
    used_set.add(root)

    depth,depth_dict,shortest_path = 0,defaultdict(int),defaultdict(int)
    depth_dict[root] = depth
    shortest_path[root] = 1

    bfs = []
    bfs.append(root)

    while len(bfs)>0:
        using_node = bfs.pop(0)
        neighbor = adj[using_node]
        used_lst.append(using_node)
        for i in neighbor:
            if i in used_set:
                if depth_dict[using_node] + 1 == depth_dict[i]:
                    shortest_path[i]+=shortest_path[using_node]
                    parent[i].add(using_node)
            else:
                bfs.append(i)
                used_set.add(i)
                parent[i].add(using_node)
                shortest_path[i]+=shortest_path[using_node]
                depth_dict[i] = depth_dict[using_node]+1

    edge_value,node_value = defaultdict(float),defaultdict(float)
    for j in used_lst:
        node_value[j]=1.0
    edge_value = edge_calculation(used_lst,parent,node_value,edge_value,shortest_path)
    return edge_value

def betweenness_calculation(vertice,adj):
    betweenness_dict = defaultdict(int)
    for node in vertice:
        edge_calculation_value = breadth_first_search(node,adj)
        for name, value in edge_calculation_value.items():
            betweenness_dict[name]+=float(value/2)
    betweenness_lst = []
    for name_use,cal_result in betweenness_dict.items():
        list_ = list(name_use)
        betweenness_lst.append(((list_[0],list_[1]),cal_result))
        betweenness_lst.sort(key = lambda x:(-x[1],x[0]))
    return betweenness_lst

betweenness = betweenness_calculation(vertice,adj)
#print(betweenness)
with open(betweenness_output_file_path,"w") as file:
    for i in betweenness:
        file.write(f"{i[0]},{round(i[1],5)}\n")



def detect_single_community(node,adj):
    adjacent,visited_node,single_com = adj[node],set(),set()
    default_value = True
    while default_value:
        visited_node = visited_node.union(adjacent)
        adj_adj_node = set()
        for i in adjacent:
            adj_adj_node = adj_adj_node.union(adj[i])
        visited_adj_adj_node = visited_node.union(adj_adj_node)
        difference = len(visited_adj_adj_node) - len(visited_node)
        if difference == 0:
            default_value = False
            break
        adjacent = adj_adj_node - visited_node

    single_com = visited_node
    if len(single_com) == 0:
        single_com = {node}
    return single_com




def detect_overall_community(node, vertice,adj):
    visited_node = detect_single_community(node,adj)
    unvisited_node = vertice - visited_node
    all_communities = []
    all_communities.append(visited_node)
    default_value = True
    default = True
    if len(unvisited_node) >0 :
        default = True
    else:
        default = False
    while default_value:
        updated_visited_nodes = detect_single_community(random.sample(unvisited_node,1)[0],adj)
        visited_node = visited_node.union(updated_visited_nodes)
        all_communities.append(updated_visited_nodes)
        unvisited_node = vertice - visited_node
        if len(unvisited_node)==0:
            default_value = False
            break
    return all_communities


def compute_modularity(all_communities,m_value,adj):
    modularity = 0.0
    for community in all_communities:
        for i in community:
            for j in community:
                actual_value = 1.0 if j in adj[i] else 0
                expected_value = (len(adj[i])*len(adj[j]))/(2*m_value)
                modularity += (actual_value-expected_value)
    modularity = modularity/(2*m_value)
    return modularity

m = len(edge)
half_edge = m/2
half_edge_copy = half_edge
assumed_modularity = -10000
while True:
    first_betweenness_value = betweenness[0][1]
    for i in betweenness:
        if i[1]==first_betweenness_value:
            adj[i[0][0]].remove(i[0][1])
            adj[i[0][1]].remove(i[0][0])
            half_edge_copy-=1

    ongoing_comm = detect_overall_community(random.sample(vertice,1)[0], vertice,adj)
    ongoing_modularity = compute_modularity(ongoing_comm,half_edge,adj)
    if ongoing_modularity>assumed_modularity:
        assumed_modularity = ongoing_modularity
        optimal_communities = ongoing_comm

    if half_edge_copy == 0:
        break

    betweenness = betweenness_calculation(vertice,adj)

final_community = sc.parallelize(optimal_communities).map(lambda i:sorted(i)).sortBy(lambda i:(len(i),i)).collect()
with open(community_output_file_path,"w") as file:
    for i in final_community:
        file.write(str(i)[1:-1]+'\n')
total_time = time.time()-start_time

print("Duration: " + str(total_time))









