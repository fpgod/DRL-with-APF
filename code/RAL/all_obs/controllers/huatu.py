import json
import numpy as np
path_data={"agent1":{},"agent2":{}}
a = np.load('a.npy')
print(a[0])
b = np.load('b.npy')
path_data['agent1']['att_x']= a.tolist()
path_data['agent2']['att_y']= b.tolist()
print(path_data)

with open("output/path_data.json","w") as f:
    json.dump(path_data,f)
    print("1")