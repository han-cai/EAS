# Reinforcement Learning for Architecture Search by Network Transformation

This is a repository of the experiment code supporting the paper **Reinforcement Learning for Architecture Search by Network Transformation**. 

The discovered top networks along with their weights are provided under the folder **model**. For checking these networks, please run under the folder of **code**:
```bash
$ python3 main.py "../model/top_net1" --test "cifar10"
```
and you will get
```bash
test acc: 0.9325921535491943
``` 
Similarly run
```bash
$ python3 main.py "../model/top_net2" --test "cifar10"
```
and you will get
```bash
test acc: 0.9430088400840759
``` 
