import numpy as np
import networkx as nx
import pandas as pd

data = pd.read_csv('routes.dat', sep=",", header=None)
routes = data[[2,4]]
routes.insert(2, "Route", routes[2]+routes[4], True)

routes.insert(2, "Count", routes.groupby('Route')['Route'].transform('count'), True)
routes = routes.drop_duplicates()
routes.sort_values(by=['Route'])

### Merge with airport info

airport_cols = ('airport_id', 'airport_name', 'city', 'country', 'iata', 'icao', 'latitude', 'longitude', 'altitude', 'timezone', 'dst',
               "Tz database time zone",'Type','source')
airports = pd.read_csv('airports.txt',  names = airport_cols,header = None)

routes = routes.merge(airports, left_on = "from", right_on = "iata") \
.merge(airports, left_on='to', right_on='iata', suffixes=('_source', '_dest'))

### Create graph

G = nx.convert_matrix.from_pandas_edgelist(routes, 2, 4, 'Count')

print(G.number_of_edges())
print(G.number_of_nodes())

# Get connected components

graphs = list(nx.connected_component_subgraphs(G))
G1 = graphs[0] ## Main component



# find where airplanes are born and which are the absorbing states
a = routes[2].drop_duplicates()
a = a.tolist()
b = routes[4].drop_duplicates()
b = b.tolist()
set(a) - set(b)
set(b) - set(a)
