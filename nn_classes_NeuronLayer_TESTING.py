from math import exp
from random import random

def sigmoid_d( x ):
    return x * (1 - x)

class Neuron():
    def __init__(self, amountOfInputs):
        self.weights = [random() for i in range(amountOfInputs)]


    def calculateOutput( self , inputs):
        res = 0
        for i in range(len(inputs)):
            res += inputs[i] * self.weights[i]
        return self.sigmoid(res)

    def sigmoid( self , x):
        return 1 / (1 + exp(-x))

    def getWeightAtIndex( self , index):
        return self.weights[index]

    def adjustWeightAtIndex( self , index, delta):
        self.weights[index] += delta

    def getTotalNumberOfWeights( self ):
        return len(self.weights)

class NeuronLayer():
    def __init__(self, amountOfInputs, amountOfNeurons):
        self.neurons = [Neuron(amountOfInputs) for i in range(amountOfNeurons)]
        self.weight_d = [[[] for j in range(amountOfInputs)] for i in range(len(self.neurons))]

    def calculateOutput( self , inputs):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.calculateOutput(inputs))
        return outputs

    def adjustWeights( self , neuronIndex, weightIndex, delta):
        self.neurons[neuronIndex].adjustWeightAtIndex(weightIndex, delta)

    def getNeruonAtIndex( self , index):
        return self.neurons[index]

    def getTotalNumberOfNeurons( self ):
        return len(self.neurons)

    def addWeightDelta( self , neuronIndex, weightIndex, delta):
        self.weight_d[neuronIndex][weightIndex].append(delta)

    def getWeightDelta( self , neuronIndex, weightIndex, index):
        return self.weight_d[neuronIndex][weightIndex][index]

    def getWeights( self ):
        res = []
        for neuronIndex in range(self.getTotalNumberOfNeurons()):
            res.append([])
            for weightIndex in range(self.getNeruonAtIndex(neuronIndex).getTotalNumberOfWeights()):
                res[-1].append(self.getNeruonAtIndex(neuronIndex).getWeightAtIndex(weightIndex))
        return res


class NeuralNetwork():
    def __init__(self, amountOfInputs, hiddenlayers, outputlayerNeurons):
        lastLayerNeurons = amountOfInputs
        self.hidden = []
        for hiddenlayerArchitecture in range(len(hiddenlayers)):
            self.hidden.append(NeuronLayer(lastLayerNeurons, hiddenlayers[hiddenlayerArchitecture]))
            lastLayerNeurons = hiddenlayers[hiddenlayerArchitecture]
        self.outputlayer = NeuronLayer(lastLayerNeurons, outputlayerNeurons)

    def predict( self , inputs):
        lastlayeroutput = inputs
        for hiddenlayer in self.hidden:
            lastlayeroutput = hiddenlayer.calculateOutput(lastlayeroutput)
        out = self.outputlayer.calculateOutput(lastlayeroutput)
        return out

    def train( self , inputs, outputs, learningrate=0.5, iterations=100000):
        for j in range(iterations):
            hl_weight_deltas = [[[[] for index2 in range(hiddenlayer.getNeruonAtIndex(index).getTotalNumberOfWeights())] for index in range(hiddenlayer.getTotalNumberOfNeurons())] for hiddenlayer in self.hidden]
            ol_weight_deltas = [[[] for index2 in range(self.outputlayer.getNeruonAtIndex(index).getTotalNumberOfWeights())] for index in range(self.outputlayer.getTotalNumberOfNeurons())] # [[[], []], [[], []]]
            for i in range(len(inputs)):
                # forward pass
                out_h = []
                lastoutput = inputs[i]
                for hiddenlayer in self.hidden:
                    out_h.append(hiddenlayer.calculateOutput(lastoutput))
                    lastoutput = out_h[-1]
                out_o = self.outputlayer.calculateOutput(lastoutput)

                error = [outputs[i][neuronIndex] - out_o[neuronIndex] for neuronIndex in range(self.outputlayer.getTotalNumberOfNeurons())]

                # calculate the deltas for the weights that are connected to the output layer
                o_deltas = [ [ ] for index in range( self.outputlayer.getTotalNumberOfNeurons() ) ]
                for neuronIndex in range( self.outputlayer.getTotalNumberOfNeurons() ):
                    for weightIndex in range( self.outputlayer.getNeruonAtIndex( neuronIndex ).getTotalNumberOfWeights() ):
                        o_deltas[ neuronIndex ].append( error[ neuronIndex ] * sigmoid_d( out_o[ neuronIndex ] ) )
                        ol_weight_deltas[neuronIndex][weightIndex].append(o_deltas[neuronIndex][weightIndex] * out_h[-1][weightIndex] * learningrate)

                # calculate the deltas for the hidden layers
                beforeLayer = self.outputlayer
                beforeDeltas = o_deltas
                for hiddenlayerIndex in reversed(range(len(self.hidden))):
                    h_current_weights = self.hidden[hiddenlayerIndex].getWeights()  # ol => [[0.9695345649941273, 0.060554043477679564], [0.7799243716936369, 0.015185710182257228]]
                    h_deltas = [[0 for index2 in range(len(h_current_weights[0]))] for index in range(len(h_current_weights))]  # [[0, 0], [0, 0]]
                    for h_neuronIndex in range( self.hidden[hiddenlayerIndex].getTotalNumberOfNeurons() ):
                        # tempSum is the error of the hiddenlayer
                        tempSum = 0
                        for o_neuronIndex in range( beforeLayer.getTotalNumberOfNeurons() ):
                            tempSum += beforeLayer.getNeruonAtIndex( o_neuronIndex ).getWeightAtIndex( h_neuronIndex ) * \
                                       beforeDeltas[ o_neuronIndex ][ h_neuronIndex ]
                        # save the weight deltas in a list to update later
                        for weightIndex in range( len( h_deltas[ h_neuronIndex ] ) ):
                            h_deltas[ h_neuronIndex ][ weightIndex ] = tempSum * sigmoid_d( out_h[hiddenlayerIndex][ h_neuronIndex ] )
                            hl_weight_deltas[hiddenlayerIndex][ h_neuronIndex ][ weightIndex ].append(
                                h_deltas[ h_neuronIndex ][ weightIndex ] * inputs[ i ][ weightIndex ] * learningrate )
                    beforeLayer = self.hidden[hiddenlayerIndex]
                    beforeDeltas = h_deltas

            # update all the weights with the saved weight deltas
            for neuronIndex in range( self.outputlayer.getTotalNumberOfNeurons() ):
                for weightIndex in range( self.outputlayer.getNeruonAtIndex( neuronIndex ).getTotalNumberOfWeights() ):
                    for d in ol_weight_deltas[ neuronIndex ][ weightIndex ]:
                        self.outputlayer.adjustWeights( neuronIndex, weightIndex, d )
            for hiddenlayerIndex in range(len(self.hidden)):
                for neuronIndex in range( self.hidden[hiddenlayerIndex].getTotalNumberOfNeurons() ):
                    for weightIndex in range( self.hidden[hiddenlayerIndex].getNeruonAtIndex( neuronIndex ).getTotalNumberOfWeights() ):
                        for d in hl_weight_deltas[hiddenlayerIndex][ neuronIndex ][ weightIndex ]:
                            self.hidden[hiddenlayerIndex].adjustWeights( neuronIndex, weightIndex, d )



if __name__ == '__main__':
    input = [[1, 1], [0, 1], [1, 0], [0, 0]]
    target = [[0, 1], [1, 0], [1, 0], [0, 1]]

    n = NeuralNetwork(2, [2], 2)
    print(n.predict(input[0]))
    n.train(input, target)
    print('')
    for i in input:
        print(n.predict(i))