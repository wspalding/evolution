import random
import uuid
import datetime

# use class instead of dict to store network data
class Network_info():
    def __init__(self, **kwargs):
        self.id = kwargs.get('id', uuid.uuid4())
        self.num_layers = kwargs.get('num_layers', 1)
        self.layer_size = kwargs.get('layer_size', 1)
        self.dropout = kwargs.get('dropout', 0.2)
        self.valid_activations = kwargs.get('valid_activations', ['relu', 'softmax', 'sigmoid', 'tanh'])
        self.activation = kwargs.get('activation', 'relu')
        self.valid_optimizers = kwargs.get('valid_optimizers',['SGD', 'Adam'])
        self.optimizer = kwargs.get('optimizer', 'SGD')
        self.starting_generation = kwargs.get('starting_generation', 0)
        self.age = kwargs.get('age', 0)
        self.accuracy = -1
        self.train_time = -1

        # TODO: improve ways that model can mutate
    def mutate(self, num_mutations):
        for i in range(num_mutations):
            attribute = random.randint(0,8)
            if attribute == 1:
                self.num_layers = max(self.num_layers + random.choice([1,-1]), 1)
            elif attribute == 2:
                self.layer_size = max(self.layer_size + random.choice([1,-1]), 1)
            elif attribute == 3:
                self.dropout = max(self.dropout + random.choice([0.01,-0.01]), 0)
            elif attribute == 4:
                o = random.randint(0,len(self.valid_optimizers)-1)
                self.optimizer = self.valid_optimizers[o]
            elif attribute == 5:
                a = random.randint(0,len(self.valid_activations)-1)
                self.activation = self.valid_activations[a]
            elif attribute == 6:
                # mutate 1 more time
                # print('called')
                i -= 2
            else:
                # dont mutate
                pass

    def __str__(self):
        return "id: {},\n\tnum_layers:{},\n\tlayer_size:{},\n\tdropout:{},\n\toptimizer:{},\n\tactivation:{},\n\tgeneration:{},\n\tage:{}".format(self.id, self.num_layers, self.layer_size, self.dropout, self.optimizer, self.activation, self.starting_generation, self.age)

    def __eq__(self, other):
        return (self.num_layers == other.num_layers) and (self.layer_size == other.layer_size) and (self.dropout == other.dropout) and (self.optimizer == other.optimizer) and (self.activation == other.activation)
