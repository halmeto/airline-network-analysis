import numpy as np
import networkx as nx
import pandas as pd

data = pd.read_csv('routes.dat', sep=",", header=None)
routes = data[[2,4]]
routes.insert(2, "Route", routes[2]+routes[4], True)

routes.insert(2, "Count", routes.groupby('Route')['Route'].transform('count'), True)
routes = routes.drop_duplicates()
routes.sort_values(by=['Route'])

G = nx.convert_matrix.from_pandas_edgelist(routes, 2, 4, 'Count')

print(G.number_of_edges())
print(G.number_of_nodes())

# find where airplanes are born and which are the absorbing states
a = routes[2].drop_duplicates()
a = a.tolist()
b = routes[4].drop_duplicates()
b = b.tolist()
set(a) - set(b)
set(b) - set(a)
