import numpy as np
import networkx as nx
import pandas as pd

data = pd.read_csv('routes.dat', sep=",", header=None)
routes = data[[2,4]]
routes.insert(2, "Route", routes[2]+routes[4], True)

routes.insert(2, "Count", routes.groupby('Route')['Route'].transform('count'), True)
routes = routes.drop_duplicates()
routes.sort_values(by=['Route'])
