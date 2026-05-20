# Day 4

# Neural Networks
It is a network thats built to learn patterns from data. It consists of smaller parts called neurons. All the neurons are interconnected. The whole network consists of multiple layers. First layer is input layer, then there are Hidden layers, and finally there is Output layer. The network learns to adjust the weights in such a way that the cost is minimum. The lesser the cost is, the more efficient it is.

# Training
Training is the process of making the model understand its task and make it more efficient with each pass. When we run it once, it has some error, the cost, when we run it the next time, we change the weights in such a way that the cost is reduced. This reduction/improvement in cost with each iteration is what is training. When do do this thousands, millions of times, thats when the model actually improves. We need to train the data multiple times for it to give some actual fruitful answer.

# Why we need Data for Training
The model needs examples to learn from. Unless we give it those examples/data, the model does not know what is right and what is wrong. For eg: We give it images of numbers 0-9 and label their answers alongwith. 
The model cant train if it has nothing to compare against. If a model makes a mistake, and its wrong, it has no way of knowing that.

# Gradient Descent
It is the optimization method we use to make the neural network better. We aim to reach the least cost in the gradient. We can reach the least cost by finding the slope at points and moving upwards or downwards from it. We move by taking small steps towards the direction of negative slope. We do this till we reach the local minima. To ensure we can also find the global minima, we use a process called SGD. Soichostic Gradient Descent.

# Backpropagation
This refers to backtracking process of a neural network. When we find an error in the cost, we go back through the multiple layers and distrubute the cause of the error in cost likewise. We do this by figuring our how much each weight contributed to the error. Then each of these weights are updated in a way such that the error is reduced in the next iteration. WIth each step, the error slightly reduces a bit. 

# Attention and Transformers
Most models before transformers focused on one word at a time, they did not have a memory to store the previous word. This did not let them develop the whole context of the sentence. It could only focus on one word at a time.

Attention refers to giving more priority to things that matter more. In a sentance, the attention is important because newer words can connect to words depending on the priority of the word. The more importance a word has, the higher its attention will be. 

Transformers were designed so that we can remember the context of the previous words better while moving onto the newer words. It goes in a sequential order. It could do this better because it ran many tokens at the same time parallely, this improved its parallelism. 

Reference: https://jalammar.github.io/illustrated-transformer/
