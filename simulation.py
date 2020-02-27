import numpy as np
import networkx as nx
import pandas as pd
import queue
import random
from collections import Counter
import matplotlib.pyplot as plt

# initialize route data
data = pd.read_csv('routes.dat', sep=",", header=None)
routes = data[[2,4]]
routes.insert(2, "Route", routes[2]+routes[4], True)
routes.insert(2, "Count", routes.groupby('Route')['Route'].transform('count'), True)
routes = routes.drop_duplicates()
#routes.sort_values(by=['Route'])
len(data)

# create graph
G = nx.convert_matrix.from_pandas_edgelist(routes, 2, 4, ['Count'])
G.number_of_edges()
G.number_of_nodes()

# initialize coordinates for plotting
coords = pd.read_csv('airports.dat', sep = ",", header = None)
coords = coords[[4, 6, 7]]
coords.columns = ["airport", "lat", "long"]
coords.set_index("airport", drop=True, inplace=True)
locs = dict()
for a in G.nodes:
    if(coords.index.contains(a)):
        locs[a] = [coords.loc[a][1], coords.loc[a][0]]
    else:     # ignore the airports that lack location information
        locs[a] = [400, 0]

# run simulations
def simulate(inf_rate = 0.1, n_iter = 100):
        Q = queue.Queue(maxsize = G.number_of_edges())
        infected = np.zeros(n_iter)
        for i in range(n_iter):
            nx.set_node_attributes(G, 0, 'infected')
            G.nodes['WUH']['infected'] = 1
            Q.put('WUH')
            while(not Q.empty()):
                next = Q.get()
                for nb in G[next]:
                    r = np.random.uniform()
                    if(r < 1-(pow(1-inf_rate, G[next][nb]['Count']))):
                        if(G.nodes[nb]['infected'] == 0):
                            Q.put(nb)
                            G.nodes[nb]['infected'] = 1
            infected[i] = sum(nx.get_node_attributes(G, 'infected').values())
        return np.mean(infected)

probs = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 0.75, 1]
n_inf = np.zeros(len(probs))

i = 0
for p in probs:
    n_inf[i] = simulate(p)
    i = i+1

n_inf
n_inf/G.number_of_nodes()
plt.plot(probs, n_inf/G.number_of_nodes())
plt.xlabel("Spreading rate p")
plt.ylabel("Proportion of infected locations")
plt.savefig("spreading_rate.png")
plt.show()

# plot the spreading of virus
def simulate_and_plot(file, title, inf_rate = 0.1, n_iter = 100):
    Q = queue.Queue(maxsize = G.number_of_edges())
    plotdata = {"airport":G.nodes(), "infected":0}
    infected = pd.DataFrame(plotdata)
    infected.set_index("airport", drop=True, inplace=True)
    infected.loc["WUH"] = 1
    for i in range(n_iter):
        nx.set_node_attributes(G, 0, 'infected')
        G.nodes['WUH']['infected'] = 1
        Q.put('WUH')
        while(not Q.empty()):
            next = Q.get()
            for nb in G[next]:
                r = np.random.uniform()
                if(r < 1-(pow(1-inf_rate, G[next][nb]['Count']))):
                    if(G.nodes[nb]['infected'] == 0):
                        Q.put(nb)
                        G.nodes[nb]['infected'] = 1
                        infected.loc[nb] = infected.loc[nb] + 1
    color_map = []
    for node in G:
        if infected["infected"].loc[node] < n_iter/4:
            color_map.append('dodgerblue')
        elif infected["infected"].loc[node] < n_iter/2:
            color_map.append('skyblue')
        elif infected["infected"].loc[node] < 3*n_iter/4:
            color_map.append('salmon')
        else:
            color_map.append('r')
    plt.figure()
    axes = plt.gca()
    axes.set_xlim([-190,190])
    axes.set_ylim([-65,85])
    plt.title(title)
    nx.draw_networkx_nodes(G, locs, nodelist=G.nodes(), node_size = 2, node_color=color_map)
    plt.savefig(file)
    plt.show()


files = ["0_05.png", "0_2.png", "0_5.png"]
titles = ["p = 0.05", "p = 0.2", "p = 0.5"]
rates = [0.05, 0.2, 0.5]

for i in range(len(rates)):
    simulate_and_plot(files[i], titles[i], rates[i], 100)


# individual level simulation
data = pd.read_csv('routes.dat', sep=",", header=None)
routes = data[[2,4]]
G2 = nx.convert_matrix.from_pandas_edgelist(routes, 2, 4)

def simulate2(days = 10, r0 = 2):
    # initialize current situation
    current = ['WUH']*65000 + ['PEK'] * 10000 + ['ICN'] * 977 + ['HND'] * 170 + ['NAP'] * 283 + ['LHR'] * 13 + ['TXL'] * 16 + ['KWI'] * 8 + ['IKA'] * 95 + ['BAH'] * 8 + ['DXB'] * 13 + ['BKK'] * 37 + ['KUL'] * 22 +['SIN'] * 90 +['HAN'] * 16 + ['TPE'] * 31 + ['HKG'] * 84 + ['SYD'] * 22 + ['LAX'] * 33 + ['DFW'] * 10 + ['BOS'] * 10 + ['YVR'] * 11 + ['CDG'] * 12
    d = {'loc': current, 'phase': np.round(np.random.uniform(0,7,len(current)))}
    sick = pd.DataFrame(d)
    print('Initialized')
    # simulate
    for i in range(days):
        sick = sick.set_index(pd.Series(range(len(sick))))
        sick['phase'] = sick['phase'] + 1
        n_fly = np.random.poisson(len(sick)/730) # each day on average every 730th person flies
        fly = np.sort(random.sample(range(len(sick)), n_fly)) # decide who fly
        for j in fly:
            dest = random.sample(list(G2[sick.iloc[j][0]]), 1)[0] # decide where everyone flies
            sick.at[j,'loc']  = dest
        n_infect = np.random.poisson(len(sick)/(7/r0)) # during their sickness of 7 days everyone infects on average r0 others
        infect = np.sort(random.sample(range(len(sick)), n_infect)) # sample those who infect others
        new_d = {'loc' : sick['loc'][infect], 'phase' : 0}
        new_df = pd.DataFrame(new_d)
        sick = sick.append(new_df)
        sick = sick[sick.phase != 7]
        print('Simulating... Day: ' +  str(i))
    return sick

# run simulation
sim = simulate2(30, 2)

# plot initial situation
current = ['WUH']*65000 + ['PEK'] * 10000 + ['ICN'] * 977 + ['HND'] * 170 + ['NAP'] * 283 + ['LHR'] * 13 + ['TXL'] * 16 + ['KWI'] * 8 + ['IKA'] * 95 + ['BAH'] * 8 + ['DXB'] * 13 + ['BKK'] * 37 + ['KUL'] * 22 +['SIN'] * 90 +['HAN'] * 16 + ['TPE'] * 31 + ['HKG'] * 84 + ['SYD'] * 22 + ['LAX'] * 33 + ['DFW'] * 10 + ['BOS'] * 10 + ['YVR'] * 11 + ['CDG'] * 12
len(current)
d = {'loc': current, 'phase': np.round(np.random.uniform(0,7,len(current)))}
sick = pd.DataFrame(d)
count1 = Counter(sick['loc'])

color_map = []
node_size = []
for node in G2:
    if count1[node] == 0:
        color_map.append('lightblue')
        node_size.append(0.1)
    elif count1[node] < 10:
        color_map.append('salmon')
        node_size.append(2)
    elif count1[node] < 100:
        color_map.append('orangered')
        node_size.append(5)
    else:
        color_map.append('red')
        node_size.append(20)
plt.figure()
axes = plt.gca()
axes.set_xlim([-190,190])
axes.set_ylim([-65,85])
plt.title("Initial situation: ~77 000 cases")
nx.draw_networkx_nodes(G2, locs, nodelist=G2.nodes(), node_size = node_size, node_color=color_map)
plt.savefig('initial.png', dpi=500)
plt.show()


# plot current situation
len(sim)
len(np.unique(sim['loc']))
title = "After 14 days (R0 = 5) : ~108 000 000 cases"
file = "r05.png"
count2 = Counter(sim['loc'])
color_map = []
node_size = []
for node in G2:
    if count2[node] == 0:
        color_map.append('lightblue')
        node_size.append(0.1)
    elif count2[node] < 10:
        color_map.append('salmon')
        node_size.append(2)
    elif count2[node] < 100:
        color_map.append('orangered')
        node_size.append(5)
    else:
        color_map.append('red')
        node_size.append(20)
plt.figure()
axes = plt.gca()
axes.set_xlim([-190,190])
axes.set_ylim([-65,85])
plt.title(title)
nx.draw_networkx_nodes(G2, locs, nodelist=G2.nodes(), node_size = node_size, node_color=color_map)
plt.savefig(file, dpi=500)
plt.show()
