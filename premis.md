# The source of the idea
evolutionary algorithms take a bunch of random models, scores them, takes the best, and has them "reporduce" with other chosen models, then repeats the process on the new set of models

when doing this it kills off all of the parents and has the new models scored,

# question
what if instead of killing off all of the parents, we kept the population of models being tested constant, and include parents (and grandparents...) when selecting for the best model.

Add in a model for optimizing the reproduction function (learns what pairs of models makes the best babies) 

and see how well it works at a given task (ms.pacman?)
