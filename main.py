

# make envoronment

population_size = 50
models = create_models(population_size) # creates 50 models
for i in range(total_generations)
    scores = [(model, score) for (model, score) in environ.run(model)]
    print(scores)
    scores.sort(score) # sort from best score to worst score
    new_models = scores[0:population_size]

    child_models = reproduce(new_models, population_size^2) # returns list of new models,
    # takes age into account when deciding which pairs should reproduce,
    # allows same parents to repoduce multiple times

    models = [new_models, child_models]


def reproduce(model_list, num_children):
    children = []
    for i in range(model_list.size()):
        for j in range(model_list.size()):
            children.append(scoring_neural_net(model_list[i], model_list[j])))
    # how to choose how many children each pair should have [0->4] as well as what fetures should be passed on
    # fetures parents are judged on:
        # model.age
        # model.score
        # model.hyperparameters
        # model.num_layers
        # ect...
    # each child is a new model that had the best combination of features as given by NN + random variations
    children = children.sort(sum(parent1.score, parent2.score))[0:num_children]
    return children
