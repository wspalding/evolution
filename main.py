from genetic_algo import Genetic_algorithm, Network_info
import train_cfar10 as cfar10
import tensorflow as tf

print("GPU Available: ", tf.test.is_gpu_available())

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

starting_population = [Network_info()]
evo = Genetic_algorithm(starting_population=starting_population,
                        num_generations=2,
                        train_epochs=1,)
final_models = evo.evolve(dataset)
# print('models: ', final_models)
# print(len(final_models))
