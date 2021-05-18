import os
import time
import sys
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from collections import defaultdict
from itertools import combinations
import graphframes
from graphframes import GraphFrame


os.environ["PYSPARK_SUBMIT_ARGS"] = ("--packages graphframes:graphframes:0.6.0-spark2.3-s_2.11")


conf = SparkConf().setAppName("553hw4").setMaster('local[*]')
sc = SparkContext(conf=conf)
sc.setLogLevel("WARN")
sql_sc = SQLContext(sc)

filter_threshold = int(sys.argv[1])
input_file_path = sys.argv[2]
community_output_file_path = sys.argv[3]

start_time = time.time()
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
                    a1.add((i[0],))
                    a1.add((j[0],))
                    a2.add(tuple((i[0],j[0])))
    return a1,a2

vertice,edge = find_same(process_data)
vertice_lst = list(vertice)
edge_lst = list(edge)

vertice_sql = sql_sc.createDataFrame(vertice_lst,['id'])
edge_sql = sql_sc.createDataFrame(edge_lst,['src','dst'])

g = GraphFrame(vertice_sql, edge_sql)
result = g.labelPropagation(maxIter=5)
result = result.rdd.map(lambda i:(i[1],i[0])).groupByKey()
result = result.map(lambda i: list(sorted(i[1]))).sortBy(lambda i:(len(i),i)).collect()



with open(community_output_file_path,'w') as file:
    for i in result:
        file.write(str(i)[1:-1])
        file.write('\n')
        

total_time = time.time()-start_time
print('Duration: ' + str(total_time))


