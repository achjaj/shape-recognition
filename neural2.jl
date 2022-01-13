using Distributions
using Random
using LinearAlgebra

generator = MersenneTwister(42)
d = Normal(0, sqrt(2/(64 + 10)))

W1 = rand(generator, d, 20, 64)
b1 = rand(generator, d, 64)

W2 = rand(generator, d, 10, 20)
b2 = rand(generator, d, 20)

function forward(x)
    hidden = tanh.(W1*x + b1)
    
    softmax(W2*hidden + b2), hidden
end

