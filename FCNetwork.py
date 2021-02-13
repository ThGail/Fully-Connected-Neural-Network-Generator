import numpy as np
import matplotlib.pyplot as plt

'''
    *** DESCRIPTION ***
    The following class generates a complete neural network, without biases.
    To initialize the network, we need to give the network's dimensions in an array (let L be this array and N be the number of layers) :
        - for each i, natural number in [0, N-1], L[i] is the number of neurons that compose the layer i 
        - i = 0 is the input layer ; i = N is the output layer 
'''

class Complete_Network :
    def __init__(self, dimension = [2,1]):

        #-- Weights initionalization -- #
        self.layerWeights = [2*np.random.random((dimension[i+1],dimension[i]))-1 for i in range(len(dimension)-1)]

        #-- Manage Graphique representation -- #
        self.ordCostGraph = []
        self.absCostGraph = []

      
    def activFunction(self,x):
        return 1/(1+np.exp(-x))

      
    def activFunctionPrime(self,x):
        #it is not really the derivative ... it is a bit rearranged 
        return x*(1-x)

   
    def forward(self, x): 
        #The next layer value is A*x (* the matrice multiplication) with A the layerWeights and x the layerValue
        layerValue = [x]
        for w in self.layerWeights :
            x = self.activFunction(np.dot(w, x))
            layerValue.append(x)
        
        return layerValue

      
    def predict(self,x):
        return self.forward(x)[-1]

      
    def BackPropagation(self, inputSet, outputSet):
        self.layersValue = self.forward(inputSet)
        self.layersErrorReversed = [outputSet - self.layersValue[-1]]
        self.layersDelta = [self.layersErrorReversed[0] * self.activFunctionPrime(self.layersValue[-1])]

        for w, v in zip(reversed(self.layerWeights), reversed(self.layersValue[:-1])):
            self.layersErrorReversed.append(np.dot(w.T, self.layersDelta[-1]))
            self.layersDelta.append(self.layersErrorReversed[-1] * self.activFunctionPrime(v))
        self.layersDelta.reverse()

        for w,d,v in zip(self.layerWeights, self.layersDelta[1:], self.layersValue[:-1]):
            w += np.dot(d, v.T)

            
    def training(self, nbTraining, inputSet, outputSet):
        #input has the dimension (a,b) : a is the nb input neurons, b is the nb of samples
        #output has the dimension (c,b) : c is the nb output neurons

        for i in range(nbTraining):
            self.BackPropagation(inputSet, outputSet)
            if i%10 == 0 :
                #print(i)
                cost = np.mean(self.layersErrorReversed[0])

        #-- Graphic plot of the error evolution --#
                self.ordCostGraph.append(cost)
                self.absCostGraph.append(i)
        plt.plot(self.absCostGraph, self.ordCostGraph)
        plt.show()
        self.ordCostGraph = []
        self.absCostGraph = []

                

            

#========================================================================#
#                            TEST PART
#========================================================================#
''' For the example, let's suppose the following rules :
        - the input is a five dimensions vector, each coefficient is in {0,1}
        - the output is 1 if the two first coef are 1, else the output is 0 
'''
#-----------------------------------------------#
trainingSet = np.array([[0,0,0,1,1],
                        [0,0,1,1,1],
                        [0,1,1,1,0],
                        [1,1,1,0,0],
                        [1,1,1,1,0],
                        [1,0,0,1,1],
                        [0,1,0,1,1]]).T

outputSet = np.array([0,0,0,1,1,0,0])

TestSet = np.array([    [1,0,1,1,1],
                        [0,1,1,0,1],
                        [0,0,1,0,1],
                        [1,1,1,0,1],
                        [1,0,1,1,1]]).T

#The expected answer is [0,0,0,1,0]
#-----------------------------------------------#


#=========== First try =========#
myNetwork1 = Complete_Network([5,10,1])           # 5 is the input vector dimension, 1 is the output dimension, 10 is an arbitraty value : nb of neurons in the hidden layer
myNetwork1.training(10000, trainingSet, outputSet)
print("Network prediction 1 :")
print(myNetwork1.predict(TestSet))

#=========== Second Try =========#
myNetwork2 = Complete_Network([5,7,4,1])          # 4 layers neural network (input and output are included)
myNetwork2.training(10000, trainingSet, outputSet)
print("Network prediction 2 :")
print(myNetwork2.predict(TestSet))


