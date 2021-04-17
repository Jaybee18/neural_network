import pygame as pg
from math import exp
from random import random
from Nebenskripte.visualize_w_nn import visualize
from Nebenskripte.progress import progressBar

def sigmoid_d( x ):
    return sigmoid( x ) * (1 - sigmoid( x ))

def sigmoid( x ):
    return 1 / (1 + exp( -x ))

class Neuron():
    def __init__( self, amountOfInputs, bias ):
        self.weights = [ random() for i in range( amountOfInputs ) ]
        self.weight_bias = random()
        self.bias = bias
        self.lastnetin = None

    def calculateOutput( self, inputs ):
        # calculates the output of this neuron based on inputs, weights and bias
        #
        # z = sum( i_i * w_i ) + bias
        # a = sigmoid( z )
        #
        # where z is the total net input
        # a is the activation/output of this neuron
        # i_i the input at index i
        # w_i the weight at index i
        # sum takes the sum for i from 0 to the amount of inputs
        res = 0 + self.bias * self.weight_bias  # = self.weight_bias
        for i in range( len( inputs ) ):
            res += inputs[ i ] * self.weights[ i ]
        self.lastnetin = res
        return sigmoid( res )

    def adjustWeightAtIndex( self, index, delta ):
        self.weights[ index ] -= delta

    def adjustBiasWeight( self, delta ):
        self.weight_bias -= delta

    def getWeightAtIndex( self, index ):
        return self.weights[ index ]

    def getTotalNumberOfWeights( self ):
        return len( self.weights )


class NeuronLayer():
    def __init__( self, amountOfInputs, amountOfNeurons ):
        self.bias = 1
        self.neurons = [ Neuron( amountOfInputs, self.bias ) for i in range( amountOfNeurons ) ]
        self.weight_d = [ [ [ ] for j in range( amountOfInputs ) ] for i in range( len( self.neurons ) ) ]
        self.lastinputs = None

    def calculateOutput( self, inputs ):
        self.lastinputs = inputs
        outputs = [ ]
        for neuron in self.neurons:
            outputs.append( neuron.calculateOutput( inputs ) )
        return outputs

    def adjustWeights( self, neuronIndex, weightIndex, delta ):
        self.neurons[ neuronIndex ].adjustWeightAtIndex( weightIndex, delta )

    def getNeruonAtIndex( self, index ):
        return self.neurons[ index ]

    def getTotalNumberOfNeurons( self ):
        return len( self.neurons )

    def addWeightDelta( self, neuronIndex, weightIndex, delta ):
        self.weight_d[ neuronIndex ][ weightIndex ].append( delta )

    def getWeightDelta( self, neuronIndex, weightIndex, index ):
        return self.weight_d[ neuronIndex ][ weightIndex ][ index ]

    def getWeights( self ):
        res = [ [ self.getNeruonAtIndex( neuronIndex ).getWeightAtIndex( weightIndex ) for weightIndex in range( self.getNeruonAtIndex( neuronIndex ).getTotalNumberOfWeights() ) ] for neuronIndex in range( self.getTotalNumberOfNeurons() ) ]
        # res = []
        # for neuronIndex in range(self.getTotalNumberOfNeurons()):
        #    res.append([])
        #    for weightIndex in range(self.getNeruonAtIndex(neuronIndex).getTotalNumberOfWeights()):
        #        res[-1].append(self.getNeruonAtIndex(neuronIndex).getWeightAtIndex(weightIndex))
        return res

    def getBiasWeights( self ):
        res = [ self.getNeruonAtIndex( neuronIndex ).weight_bias for neuronIndex in range(self.getTotalNumberOfNeurons())]
        return res

    def getLastInputs( self ):
        return self.lastinputs


class NeuralNetwork():
    def __init__( self, amountOfInputs, hiddenlayers, outputlayerNeurons ):
        lastLayerNeurons = amountOfInputs
        self.hidden = [ ]
        for hiddenlayerArchitecture in range( len( hiddenlayers ) ):
            self.hidden.append( NeuronLayer( lastLayerNeurons, hiddenlayers[ hiddenlayerArchitecture ] ) )
            lastLayerNeurons = hiddenlayers[ hiddenlayerArchitecture ]
        self.outputlayer = NeuronLayer( lastLayerNeurons, outputlayerNeurons )

    def getWeights( self ):
        res = [ self.hidden[ i ].getWeights() for i in range( len( self.hidden ) ) ]
        res.append( self.outputlayer.getWeights() )
        return res

    def getBiasWeights( self ):
        res = [ self.hidden[ i ].getBiasWeights() for i in range(len(self.hidden)) ]
        res.append( self.outputlayer.getBiasWeights() )
        return res

    def predict( self, inputs ):
        lastlayeroutput = inputs
        for hiddenlayer in self.hidden:
            lastlayeroutput = hiddenlayer.calculateOutput( lastlayeroutput )
        out = self.outputlayer.calculateOutput( lastlayeroutput )
        return out

    def drawVisualization( self ):
        for i in range( int( self.input_range[ 0 ] / self.step_size ) ):
            for j in range( int( self.input_range[ 1 ] / self.step_size ) ):
                res = self.predict( [ (i * self.step_size) + self.start[ 0 ], (j * self.step_size) + self.start[ 1 ] ] )[ 0 ]
                if not self.fade:
                    res = round( res )
                color = (int( 255 - res * 255 ), int( res * 255 ), 0)
                pg.draw.rect(self.screen, color, pg.Rect(j * self.circle_rad, 600-self.circle_rad-i * self.circle_rad, self.circle_rad, self.circle_rad))
        pg.display.update()
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                try:
                    exit()
                except:
                    pass

    def train( self, inputs, outputs, learningrate=0.5, iterations=100_000, live_visualization=False):
        # setup for live visualization
        if live_visualization:
            pg.init()
            self.screen = pg.display.set_mode((600, 600))
            self.input_range = (1, 1)
            self.start = (0, 0)
            self.step_size = 0.01   # manually adjustable (variable mentioned in the README.txt)
            self.circle_rad = 600/(self.input_range[ 0 ] / self.step_size)
            self.fade = True        # manually adjustable (variable mentioned in the README.txt)
        progress = progressBar(inPercent=False, maxValue=iterations)
        for j in range( iterations ):
            hl_weight_deltas = [ [ [ 0 for index2 in range( hiddenlayer.getNeruonAtIndex( index ).getTotalNumberOfWeights() ) ] for index in range( hiddenlayer.getTotalNumberOfNeurons() ) ] for hiddenlayer in self.hidden ]
            ol_weight_deltas = [ [ 0 for index2 in range( self.outputlayer.getNeruonAtIndex( index ).getTotalNumberOfWeights() ) ] for index in range( self.outputlayer.getTotalNumberOfNeurons() ) ]
            for i in range( len( inputs ) ):
                # forward pass
                out_h = [ ]
                lastoutput = inputs[ i ]
                for hiddenlayer in self.hidden:
                    out_h.append( hiddenlayer.calculateOutput( lastoutput ) )
                    lastoutput = out_h[ -1 ]
                if out_h == [ ]:
                    out_h = [ inputs[ i ] ]
                out_o = self.outputlayer.calculateOutput( lastoutput )

                # calculate the MSE derivative
                error = [ -2*(outputs[ i ][ neuronIndex ] - out_o[ neuronIndex ]) for neuronIndex in range( self.outputlayer.getTotalNumberOfNeurons() ) ]

                # calculate the deltas for the weights that are connected to the output layer
                #
                # ∂e/∂w = ∂e/∂a * ∂a/∂z * ∂z/∂w * η
                #
                # where:
                # e is the totalError,
                # w is the weight,
                # a is the activation (the output of the neuron)
                # z is the netInput
                # η is the learning rate
                o_deltas = [ [ ] for index in range( self.outputlayer.getTotalNumberOfNeurons() ) ]
                for neuronIndex in range( self.outputlayer.getTotalNumberOfNeurons() ):
                    for weightIndex in range( self.outputlayer.getNeruonAtIndex( neuronIndex ).getTotalNumberOfWeights() ):
                        # calculate the total error wrt the net input and save it for late use
                        # when calculating the hidden layer weight deltas
                        #
                        # ∂e/∂z = ∂e/∂a * ∂a/∂z                     [1]
                        o_deltas[ neuronIndex ].append( error[ neuronIndex ] * sigmoid_d( self.outputlayer.getNeruonAtIndex(neuronIndex).lastnetin) )
                        # calculate the weight delta
                        #
                        # ∂e/∂w = ∂e/∂z * ∂z/∂w * η
                        ol_weight_deltas[ neuronIndex ][ weightIndex ] += o_deltas[ neuronIndex ][ weightIndex ] * out_h[ -1 ][ weightIndex ] * learningrate
                    # the bias weight is updated like this:
                    #
                    # ∂e/∂b = ∂e/∂a * ∂a/∂z * ∂z/∂b * η
                    #       = ∂e/∂z * ∂z/∂b * η
                    #       = ∂e/∂z * η          | since "∂z/∂b" is always 1
                    #
                    # because the bias is only connected to the current
                    # layer and therefore not used in any further
                    # calculations it can be updated instantly
                    self.outputlayer.getNeruonAtIndex( neuronIndex ).adjustBiasWeight( o_deltas[ neuronIndex ][ 0 ] * 1 * learningrate )

                # calculate the deltas for the hidden layers
                #
                # ∂e/∂w = ∂e/∂a_h * ∂a/∂z * ∂z/∂w * η
                # ∂e/∂a_h = ∂e/∂a_o1 + ... + ∂e/∂a_oN
                beforeLayer = self.outputlayer
                beforeDeltas = o_deltas
                for hiddenlayerIndex in reversed( range( len( self.hidden ) ) ):
                    h_current_weights = self.hidden[ hiddenlayerIndex ].getWeights()
                    h_deltas = [ [ 0 for index2 in range( len( h_current_weights[ 0 ] ) ) ] for index in range( len( h_current_weights ) ) ]
                    for h_neuronIndex in range( self.hidden[ hiddenlayerIndex ].getTotalNumberOfNeurons() ):
                        # tempSum is the totalError wrt the activation of the current hidden layer neuron
                        # ∂e/∂a_h = ∂e/∂a_o1 + ... + ∂e/∂a_oN
                        # ∂e/∂a_o1 = ∂e/a_o1 * ∂a_o1/∂z_o1 * ∂z_o1/∂a_h
                        #           |_____________________|     ||
                        #                      \/               ||
                        # this was computed before (comment in line 181 [1] and later in line 233 [2])
                        #                                       ||         so it can just be reused
                        #                                       ||
                        #                                       \/
                        # and this is just the weight connecting that previous
                        # layer neuron to the current hidden layer neuron
                        #
                        # "_oN" is used to indicate neurons of the previous layer (to the right)
                        # in the first iteration that will be the output layer, in next
                        # the hidden layer and so on
                        tempSum = 0
                        for o_neuronIndex in range( beforeLayer.getTotalNumberOfNeurons() ):
                            tempSum += beforeDeltas[ o_neuronIndex ][ h_neuronIndex ] * beforeLayer.getNeruonAtIndex( o_neuronIndex ).getWeightAtIndex( h_neuronIndex )
                        # save the weight deltas in a list to update later
                        # because the old weights are used in calculations
                        # for the next layers
                        for weightIndex in range( len( h_deltas[ h_neuronIndex ] ) ):
                            # this is saved to reuse it in next hidden
                            # layer weight updates to reduce runtime
                            #
                            # ∂e/∂z = ∂e/∂a_h * ∂a_h/∂z     [2]
                            h_deltas[ h_neuronIndex ][ weightIndex ] = tempSum * sigmoid_d( self.hidden[hiddenlayerIndex].getNeruonAtIndex(h_neuronIndex).lastnetin )
                            # ∂e/∂w = ∂e/∂z * ∂z/∂w * η
                            hl_weight_deltas[ hiddenlayerIndex ][ h_neuronIndex ][ weightIndex ] += h_deltas[ h_neuronIndex ][ weightIndex ] * self.hidden[ hiddenlayerIndex ].getLastInputs()[ weightIndex ] * learningrate
                        # ∂e/∂b = ∂e/∂a * ∂a/∂z * ∂z/∂b * η
                        self.hidden[ hiddenlayerIndex ].getNeruonAtIndex( h_neuronIndex ).adjustBiasWeight( h_deltas[ h_neuronIndex ][ 0 ] * 1 * learningrate )
                    beforeLayer = self.hidden[ hiddenlayerIndex ]
                    beforeDeltas = h_deltas

            # update all the weights with the saved weight deltas
            for neuronIndex in range( self.outputlayer.getTotalNumberOfNeurons() ):
                for weightIndex in range( self.outputlayer.getNeruonAtIndex( neuronIndex ).getTotalNumberOfWeights() ):
                    self.outputlayer.adjustWeights( neuronIndex, weightIndex, ol_weight_deltas[ neuronIndex ][ weightIndex ] / len(inputs) )
            for hiddenlayerIndex in range( len( self.hidden ) ):
                for neuronIndex in range( self.hidden[ hiddenlayerIndex ].getTotalNumberOfNeurons() ):
                    for weightIndex in range( self.hidden[ hiddenlayerIndex ].getNeruonAtIndex( neuronIndex ).getTotalNumberOfWeights() ):
                        self.hidden[ hiddenlayerIndex ].adjustWeights( neuronIndex, weightIndex, hl_weight_deltas[ hiddenlayerIndex ][ neuronIndex ][ weightIndex] / len(inputs))
            progress.update()
            if j % 100 == 0 and live_visualization:
                self.drawVisualization()
        pg.quit()


if __name__ == '__main__':
    input = [ [ 1, 1 ], [ 0, 1 ], [ 1, 0 ], [ 0, 0 ] ]
    target = [ [ 0 ], [ 1 ], [ 1 ], [ 0 ] ]

    n = NeuralNetwork( 2, [ 2 ], 1 )
    n.train( input, target, iterations=100000)
    for i in input:
        print( n.predict( i ) )

    print( n.getWeights() )
    print( n.getBiasWeights() )
    visualize( n )
