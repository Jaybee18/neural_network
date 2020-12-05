# neural_network
In this repository i will upload my neural network project

The NeuralNetwork class is the main class one should use.
It is initialised like so:
nn = NeuralNetwork(numberOfInputs, list_for_hiddenlayer_architecture, numberOfOutputs)
where "list_for_hiddenlayer_architecture" is a list/array/tuple of integers that describe the number of neurons per layer e.g.:
[2, 3]
would be a Neural Network with 2 hidden layers. The first one has 2 and the second 3 Neurons.


It is trained like so:
The input list/array/tuple is formatted like this:
[ [ inputForFirstNeuronInInputLayer, inputForSecondNeuronInInputLayer, ... ] , ...] 
And the output list/array/tuple is formatted like this:
[ [ outputForFirstNeuronInOutputLayer, outputForSecondNeuronInOutputLayer, ... ] , ...] 

nn.train(inputs, outputs)
With two optional parameters : learningrate (double) and iterations (integer)


It gives predictions like the following:
nn.predict( [ inputForFirstNeuronInInputLayer, inputForSecondNeuronInInputLayer, ... ] )
which returns a list of outputs containing the output for every output neuron

Bugs:
Currently the network can not be initialized without hidden layers
