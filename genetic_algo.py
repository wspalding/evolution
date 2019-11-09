import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from network import Network_info
import operator
import pandas as pd
import random
import uuid
import datetime


class Genetic_algorithm():
    def __init__(self, **kwargs):
        self.max_poulation = kwargs.get('max_poulation', 50)
        self.num_generations = kwargs.get('num_generations', 20)
        self.current_generation = 0
        self.population = kwargs.get('starting_population', [Network_info()])
        self.max_age = kwargs.get('max_age', self.num_generations)
        self.train_epochs = kwargs.get('train_epochs', 1000)
        self.num_children = kwargs.get('num_children', 2)
        self.num_mutations = kwargs.get('num_mutations', 1)
        self.learning_threshold = kwargs.get('learning_threshold', 1)

        # save results in pandas dataframe
        self.columns = {'network_id':[] ,'network_score(acc)':[], 'network_info':[], 'time_to_train(ms)':[], 'starting_generation':[], 'final_age':[]}
        self.results = pd.DataFrame(self.columns)

        self.save_file = kwargs.get('save_file', 'results/results_{}.csv'.format(datetime.datetime.now()))
        self.save_data = kwargs.get('save_data', True)

    def train_and_score(self, dataset):
        rankings = []
        self.current_generation += 1
        for model_dict in self.population:
            model = self.make_network_from_dict(model_dict, dataset['input_shape'], dataset['num_classes'])
            start = datetime.datetime.now()
            if dataset['name'] == 'cifar10':
                model.fit(dataset['x_train'], dataset['y_train'],
                batch_size=dataset['batch_size'],
                epochs=self.train_epochs,  # using early stopping, so no real limit
                verbose=0,
                callbacks=[EarlyStopping(patience=5)],
                validation_data=(dataset['x_test'], dataset['y_test']))
                score = model.evaluate(dataset['x_test'], dataset['y_test'], verbose=0)
                model_dict.accuracy = score[1]
                rankings.append(model_dict) # 1 is accuracy. 0 is loss.
                print ("model: {},\nscore:{}\n".format(model_dict, score))
            else: # other datasets go here
            # TODO: implament passing scoring function to train & score
                pass
            end = datetime.datetime.now()
            delta = end - start
            model_dict.train_time = int(delta.total_seconds() * 1000)
            model_dict.age += 1 # incrament age
        # rankings.sort(key=operator.itemgetter(1))
        return rankings

    def make_network_from_dict(self, network_info, input_shape, output_shape):
        model = Sequential()
        # input layer
        model.add(Dense(network_info.layer_size, activation=network_info.activation, input_shape=input_shape))
        # add layers from network dict
        for i in range(1, network_info.num_layers): # input layer is layer 0
            model.add(Dense(network_info.layer_size, activation=network_info.activation))

        model.add(Dropout(network_info.dropout))
        # output layer
        model.add(Dense(output_shape, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=network_info.optimizer,
                  metrics=['accuracy'])
        return model

    def cull(self, rankings, **kwargs):
        save = kwargs.get('save', self.save_data)
        within_age = []
        for i in rankings:
            if i.age <= self.max_age and i.accuracy >= 0:
                within_age.append(i)
            else:
                if self.save_data:
                    self.save_network_info_to_dataframe(i)

        within_age.sort(key=lambda x: x.accuracy, reverse=True)
        self.population = within_age[0:self.max_poulation]
        for i in within_age[50:]:
            if self.save_data:
                self.save_network_info_to_dataframe(i)

        return self.population

    def create_children(self):
        all_children = []
        for model in self.population:
            model2 = self.population[random.randint(0,len(self.population)-1)]
            all_children.extend(self.reproduce(model, model2))
        return all_children

    def reproduce(self, parent1, parent2):
        children = []
        for _ in range(self.num_children):
            child = Network_info(
                num_layers=random.choice([parent1.num_layers, parent2.num_layers]),
                layer_size=random.choice([parent1.layer_size, parent2.layer_size]),
                dropout=random.choice([parent1.dropout, parent2.dropout]),
                activation=random.choice([parent1.activation, parent2.activation]),
                optimizer=random.choice([parent1.optimizer, parent2.optimizer]),
                starting_generation=self.current_generation,
            )
            child.valid_optimizers = parent1.valid_optimizers
            child.valid_activations = parent2.valid_activations
            while(self.is_duplicate(child)):
                child.mutate(self.num_mutations)
            children.append(child)
        return children

    def is_duplicate(self, network_info):
        for p in self.population:
            if p == network_info:
                # print("made duplicate")
                return True
        return False

    # run whole process
    def evolve(self, dataset):
        for i in range(self.num_generations):
            print('generation: ', i, '\npopulation size: ', len(self.population), '\n')
            ranks = self.train_and_score(dataset)
            done = False
            for m in rankes:
                if m.accuracy >= self.learning_threshold:
                    done = True
                    break
            if done:
                break
            self.cull(ranks)
            new_children = self.create_children()
            self.population.extend(new_children)
        if self.save_data:
            self.save_to_file()
            print(self.results.head())
        return self.population

    def save_to_file(self):
        for model in self.population:
            if model.train_time > 0:
                self.save_network_info_to_dataframe(model)
        self.results.sort_values(by=['network_score(acc)'])
        self.results.to_csv(self.save_file)

    def save_network_info_to_dataframe(self, network):
        self.results = self.results.append({
            'network_id': network.id,
            'network_score(acc)': network.accuracy,
            'network_info': [
                network.num_layers,
                network.layer_size,
                network.dropout,
                network.activation,
                network.optimizer,
                ],
            'time_to_train(ms)': network.train_time,
            'starting_generation': network.starting_generation,
            'final_age': network.age,
            }, ignore_index=True)
