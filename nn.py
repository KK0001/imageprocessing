import numpy
import scipy.special

# nueral network
class nueralNetwork:
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        # nodes
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.unodes = outputnodes

        # weight
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5),(self.hnodes, self.inodes)) # between input and hidden
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5),(self.onodes, self.hnodes)) # between hidden and output

        # learning rate
        self.lr = learningrate

        # activation function
        self.activation_function = lambda x : scipy.special.expit(x)
        pass

    def train():
        
        pass
    
    def query(self,inputs_list):
        
        # 入力リストを行列に変換
        inputs = numpy.array(inputs_list,ndmin=2).T
        
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


input_nodes = 3
hidden_nodes = 3
output_nodes = 3
learning_rate = 0.3

n = nueralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)