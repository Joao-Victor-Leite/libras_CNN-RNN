'''
@author Lucas Lacerda
@date 05/2019
'''
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import LeakyReLU

class Convolucao(object):
    @staticmethod
    def build(width, height, channels, classes):
        """
        :param width: Largura em pixel da imagem.
        :param height: Altura em pixel da imagem.
        :param channels: Quantidade de canais da imagem.
        :param classes: Quantidade de classes para o output.

        :return: CNN com a arquitetura:
            INPUT => CONV => POOL => CONV => POOL => CONV => POOL => FC => FC => OUTPUT
        """
        inputShape = (height, width, channels)

        '''
        There are three ways to create Keras models:

        The Sequential model, which is very straightforward (a simple list of layers), but is limited to single-input, single-output stacks of layers (as the name gives away).
        The Functional API, which is an easy-to-use, fully-featured API that supports arbitrary model architectures. For most people and most use cases, this is what you should be using. This is the Keras "industry strength" model.
        Model subclassing, where you implement everything from scratch on your own. Use this if you have complex, out-of-the-box research use cases.
        '''

        #Sequential API is the easiest model to build and run in Keras. A sequential model allows us to create models layer by layer in a step by step fashion.
        #In Functional model, part or all of the inputs directly connected to the output layer. This architecture makes it possible for the neural network to learn both deep patterns and simple rules.

        '''
        A sequential model, as the name suggests, allows you to create models layer-by-layer in a step-by-step fashion.

        Keras Sequential API is by far the easiest way to get up and running with Keras, but it’s also the most limited — you cannot create models that:

        Share layers
        Have branches (at least not easily)
        Have multiple inputs
        Have multiple outputs
        Examples of seminal sequential architectures that you may have already used or implemented include:

        LeNet
        AlexNet
        VGGNet
        '''

        '''
        Once you’ve had some practice implementing a few basic neural network architectures using Keras’ Sequential API, you’ll then want to gain experience working with the Functional API.

        Keras’ Functional API is easy to use and is typically favored by most deep learning practitioners who use the Keras deep learning library.

        Using the Functional API you can:

        Create more complex models.
        Have multiple inputs and multiple outputs.
        Easily define branches in your architectures (ex., an Inception block, ResNet block, etc.).
        Design directed acyclic graphs (DAGs).
        Easily share layers inside the architecture.
        Furthermore, any Sequential model can be implemented using Keras’ Functional API.

        Examples of models that have Functional characteristics (such as layer branching) include:

        ResNet
        GoogLeNet/Inception
        Xception
        SqueezeNet
        '''

        '''INPUT SHAPE
        What flows between layers are tensors. Tensors can be seen as matrices, with shapes.

        In Keras, the input layer itself is not a layer, but a tensor. It's the starting tensor you send to the first hidden layer. This tensor must have the same shape as your training data.

        Example: if you have 30 images of 50x50 pixels in RGB (3 channels), the shape of your input data is (30,50,50,3). Then your input layer tensor, must have this shape (see details in the "shapes in keras" section).
        '''

        model = Sequential()

        #2D convolution layer (e.g. spatial convolution over images).
        #The first required Conv2D parameter is the number of filters that the convolutional layer will learn.
        model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'same', input_shape = inputShape))

        #With a Leaky ReLU (LReLU), you won’t face the “dead ReLU” (or “dying ReLU”) problem which happens when your ReLU always have values under 0 - this completely blocks learning in the ReLU because of gradients of 0 in the negative part.

        #"Unfortunately, ReLU units can be fragile during training and can "die". For example, a large gradient flowing through a ReLU neuron could cause the weights to update in such a way that the neuron will never activate on any datapoint again. If this happens, then the gradient flowing through the unit will forever be zero from that point on. That is, the ReLU units can irreversibly die during training since they can get knocked off the data manifold. For example, you may find that as much as 40% of your network can be "dead" (i.e. neurons that never activate across the entire training dataset) if the learning rate is set too high. With a proper setting of the learning rate this is less frequently an issue."

        '''
        A "dead" ReLU always outputs the same value (zero as it happens, but that is not important) for any input. Probably this is arrived at by learning a large negative bias term for its weights.

        In turn, that means that it takes no role in discriminating between inputs. For classification, you could visualise this as a decision plane outside of all possible input data.

        Once a ReLU ends up in this state, it is unlikely to recover, because the function gradient at 0 is also 0, so gradient descent learning will not alter the weights. "Leaky" ReLUs with a small positive gradient for negative inputs (y=0.01x when x < 0 say) are one attempt to address this issue and give a chance to recover.

        The sigmoid and tanh neurons can suffer from similar problems as their values saturate, but there is always at least a small gradient allowing them to recover in the long term.
        '''

        '''
        The main advantage of ReLU is that it outputs 0 and 1 thus solves the problem of Vanishing Gradient(because we don’t have to multiply extremely small values during Backpropagation). However, it has it’s own downside too. Because it outputs 0 for every negative value, a ReLU neuron might get stuck in the negative side and always output 0, and it is unlikely for it to recover. This is called as the dying ReLU problem. This is a serious problem because if a neuron is dead, then it basically learns nothing. Because of this problem, there might be the case of a large part of the network doing nothing.
        '''

        '''
        So what are the alternatives to ReLU? We can always use tanh and Sigmoid activations. Using a modified version of ReLU called Leaky ReLU, can also help get around the problem. However, in the particular example created for this experiment, all the mentioned activations fail because they suffer from Vanishing Gradients. When we consider the tradeoff between Vanishing gradients and Dying ReLU, it’s always better to have something than nothing. In Vanishing gradients, there is some learning, but in the case of dead ReLU there is no learning, the learning is halted.
        '''

        '''
        What it does? Simple – take a look at the definition from the API docs: f(x) = alpha * x for x < 0, f(x) = x for x >= 0 .

        Alpha is the slope of the curve for all x<0.
        '''
        model.add(LeakyReLU(alpha=0.1))

        #pool_size: integer or tuple of 2 integers, window size over which to take the maximum. (2, 2) will take the max value over a 2x2 pooling window. If only one integer is specified, the same window length will be used for both dimensions.#
        model.add(MaxPooling2D((2,2)))

        model.add(Conv2D(filters = 32, kernel_size = (3,3)))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D((2,2)))

        #Layers early in the network architecture (i.e., closer to the actual input image) learn fewer convolutional filters while layers deeper in the network (i.e., closer to the output predictions) will learn more filters.

        ''' FILTERS
        In every layer filters are there to capture patterns. For example in the first layer filters capture patterns like edges, corners, dots etc. In the subsequent layers we combine those patterns to make bigger patterns. Like combine edges to make squares, circle etc.

        Now as we move forward in the layers the patterns gets more complex, hence larger combinations of patterns to capture. That's why we increase filter size in the subsequent layers to capture as many combinations as possible.
        '''

        ''' KERNEL_SIZE
        The second required parameter you need to provide to the Keras Conv2D class is the kernel_size , a 2-tuple specifying the width and height of the 2D convolution window.

        The kernel_size must be an odd integer as well.

        Typical values for kernel_size include: (1, 1) , (3, 3) , (5, 5) , (7, 7) . It’s rare to see kernel sizes larger than 7×7.
        '''

        '''PADDING
        Same padding means the size of output feature-maps are the same as the input feature-maps (under the assumption of  stride=1 ). For instance, if input is  nin  channels with feature-maps of size  28×28 , then in the output you expect to get  nout  feature maps each of size  28×28  as well.
        '''

        model.add(Conv2D(filters = 64, kernel_size = (3,3)))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D((2,2)))

        '''
        A fully connected layer also known as the dense layer, in which the results of the convolutional layers are fed through one or more neural layers to generate a prediction.

        Suppose you’re using a Convolutional Neural Network whose initial layers are Convolution and Pooling layers. They layers have multidimensional tensors as their outputs. If you wanted to use a Dense(a fully connected layer) after your convolution layers, you would need to ‘unstack’ all this multidimensional tensor into a very long 1D tensor. You can achieve this using Flatten.

        A dense layer is just a regular layer of neurons in a neural network. Each neuron recieves input from all the neurons in the previous layer, thus densely connected. The layer has a weight matrix W, a bias vector b, and the activations of previous layer a.
        '''
        model.add(Flatten())

        '''
        There is no known way to determine a good network structure evaluating the number of inputs or outputs. It relies on the number of training examples, batch size, number of epochs, basically, in every significant parameter of the network.

        Moreover, a high number of units can introduce problems like overfitting and exploding gradient problems. On the other side, a lower number of units can cause a model to have high bias and low accuracy values. Once again, it depends on the size of data used for training.

        Sadly it is trying some different values that give you the best adjustments. You may choose the combination that gives you the lowest loss and validation loss values, as well as the best accuracy for your dataset, as said in the previous post.

        You could do some proportion on your number of units value
        '''
        model.add(Dense(256, activation = 'relu'))
        model.add(Dropout(0.5))

        '''
        The elements of the output vector are in range (0, 1) and sum to 1.

        Each vector is handled independently. The axis argument sets which axis of the input the function is applied along.

        Softmax is often used as the activation for the last layer of a classification network because the result could be interpreted as a probability distribution.

        The softmax of each vector x is calculated by exp(x)/tf.reduce_sum(exp(x)). The input values in are the log-odds of the resulting probability.
        '''

        #If we take an input of [1, 2, 3, 4, 1, 2, 3], the softmax of that is [0.024, 0.064, 0.175, 0.475, 0.024, 0.064, 0.175]. The output has most of its weight where the '4' was in the original input. This is what the function is normally used for: to highlight the largest values and suppress values which are significantly below the maximum value. But note: softmax is not scale invariant, so if the input were [0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3] (which sums to 1.6) the softmax would be [0.125, 0.138, 0.153, 0.169, 0.125, 0.138, 0.153]. This shows that for values between 0 and 1 softmax, in fact, de-emphasizes the maximum value (note that 0.169 is not only less than 0.475, it is also less than the initial proportion of 0.4/1.6=0.25).

        #Softmax turns arbitrary real values into probabilities, which are often useful in Machine Learning.
        model.add(Dense(classes, activation = 'softmax'))

        return model
