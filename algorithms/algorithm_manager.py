import inspect

from importlib import import_module
import re

from algorithms import hand_position
def get_all_algorithm():
    algos = []
    with open("algorithms/algorithm_order","r") as f:
        for file_name in f.read().split("\n"):
            exec(f'from algorithms import {file_name}')
            v = eval(file_name)
            #print(v)
            for name, obj in inspect.getmembers(v):
                #print(name)
                if re.search('ATC$',name):
                    algos.append(obj())
    #print(algos)
    return algos

