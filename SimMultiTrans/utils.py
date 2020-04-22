import os
import json
import pandas as pd
import numpy as np


def generate_graph_file(gps_file, aam_file):

    aam = pd.read_csv(aam_file, index_col=0, header=0)
    aam = aam.loc[(aam.index.isin(NODES))]
    aam = aam.loc[:, [str(x) for x in NODES]]
    gps = pd.read_csv(gps_file, index_col=2, header=0)
    graph = dict()
    for n in NODES:
        graph[str(n)] = dict()
        graph[str(n)]['lat'] = gps.loc[n, 'y']
        graph[str(n)]['lon'] = gps.loc[n, 'x']
        graph[str(n)]['mode'] = 'taxi'
        graph[str(n)]['nei'] = dict()
        for d in NODES:
            graph[str(n)]['nei'][str(d)] = dict()
            graph[str(n)]['nei'][str(d)]['mode'] = 'taxi'
            graph[str(n)]['nei'][str(d)]['time'] = 0
            graph[str(n)]['nei'][str(d)]['dist'] = 0
            graph[str(n)]['nei'][str(d)]['rate'] = 1/aam.loc[n, str(d)] if pd.notna(aam.loc[n, str(d)]) else 0

    return graph


def update_vehicle_initial_distribution(dist):
    with open(os.path.join(CONFIG, 'vehicle_nyc.json'), 'r') as f:
        vehicle_file = json.load(f)

    vehicle_file['taxi']['distrib'].clear()
    for idx, n in enumerate(NODES):
        vehicle_file['taxi']['distrib'][str(n)] = dist[idx]

    with open(os.path.join(CONFIG, 'vehicle_nyc.json'), 'w') as f:
        json.dump(vehicle_file, f, indent=4)


if __name__ == '__main__':

    # NODES = sorted(pd.read_csv(os.path.join(CONFIG, 'aam.csv'), index_col=0, header=0).index.values.tolist())
    # NODES = sorted([236, 237, 186, 170, 141, 162, 140, 238, 142, 229, 239, 48, 161, 107, 263, 262, 234, 68, 100, 143])
    NODES = sorted([236, 237, 186, 170, 141])

    j = json.dumps(generate_graph_file(os.path.join(CONFIG, 'gps.csv'),
                                       os.path.join(CONFIG, 'aam.csv')),
                   indent=4)
    with open(os.path.join(CONFIG, 'city_nyc.json'), 'w') as ff:
        json.dump(generate_graph_file(os.path.join(CONFIG, 'gps.csv'), os.path.join(CONFIG, 'aam.csv')),
                  ff,
                  indent=4)

    update_vehicle_initial_distribution([int(20) for i in range(len(NODES))])
