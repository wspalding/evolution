# evolution

## how to use:

    nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test = cfar10.get_cifar10()
    dataset = {
            'name': 'cifar10',
            'num_classes': nb_classes,
            'batch_size': batch_size,
            'input_shape': input_shape,
            'x_train': x_train,
            'x_test': x_test,
            'y_train': y_train,
            'y_test': y_test,
            }

    evo = Genetic_algorithm(num_generations=100,
                            train_epochs=10000,
                            max_poulation=10,
                            learning_threshold=0.8)
    final_models = evo.evolve(dataset)

## key word args:
max_poulation = number of surviving models after each generation, default 50

num_generations = number of generation run, default 20

starting_population = array of Network_info objects to start with, default [Network_info()]

max_age = the most number of generations a model can survive before being removed from population, default
num_generations

train_epochs = number of epochs that each model is trained for, default 1000

num_children = number of children each pair of models produces, default 2

num_mutations = number of mutations each child has before being added to population, default 1

learning_threshold = accuracy for any model to get to end evolution, default 1

save_file = path and file of where to save results file, default results/results_{}.csv'.format(datetime.datetime.now())

save_data = whether or not to save data to file, default true


## Network_info

### key word args

id: default uuid.uuid4())

num_layers: default 1

layer_size: default 1

dropout: default 0.2

valid_activations: default ['relu', 'softmax', 'sigmoid', 'tanh']

activation: default 'relu'

valid_optimizers: default ['SGD', 'Adam']

optimizer: default 'SGD'

starting_generation: default 0

age: default 0
