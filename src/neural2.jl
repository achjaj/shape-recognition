using Random
using Plots
import Distributions
include("functions.jl")

mutable struct Net
    weights::Vector{Matrix}
    biases::Vector{Vector}
    f::Vector{Function}
    
    function Net(sizes::Int...)
        dist = Distributions.Uniform(-0.1, 0.1)

        weights = Vector{Matrix}()
        biases = Vector{Vector}()

        for i in 1:length(sizes)-1
            w = rand(dist, sizes[i+1], sizes[i])
            b = rand(dist, sizes[i+1])

            push!(weights, w)
            push!(biases, b)
        end

        new(weights, biases, [fill(vtanh, length(sizes) - 2); softmax])
    end
end

mean(x) = sum(x)/length(x)
relu(x::Vector) = max.(x, 0)

function softmax(x::Vector)
    m = maximum(x)
    s = exp.(x .- m)
    s/sum(s)
end

function chop(v::Vector, s::Int)
    res = Vector{Vector}()
    l = length(v)

    for i in 1:s:l
        stop = i+s-1
        stop = stop <= l ? stop : l

        push!(res, v[i:stop])
    end

    res
end

function forward(net::Net, x::AbstractVector)
    outputs = Vector{AbstractVector}()
    push!(outputs, x)

    for (W, b, f) in zip(net.weights, net.biases, net.f)
        raw = W*outputs[end] + b
        push!(outputs, f(raw))
    end

    outputs
end

predict(net::Net, x::AbstractVector) = argmax(forward(net, x)[end])

function accuracy(net, x::Vector, t::Vector)
    predictions = [predict(net, i) for i in x]
    cmp = isequal.(predictions, t)
    sum(cmp)/length(x)
end

function update_weights(net::Net, batch_x::Vector, batch_t::Vector, α::Number, regularization::Number)
    out_grad(o, t) = o - [Int(==(i, t)) for i in 1:length(o)]
    hid_grad(W, g, h) = dtanh(h) * (W' * g)
    dW(g, h) = g*h'

    outputs = map(x -> forward(net, x), batch_x)
    layers = collect(zip(outputs...))
    output_layer_gradient = out_grad.(layers[end], batch_t)

    net.biases[end] -= α*mean(output_layer_gradient)
    net.weights[end] -= α*(mean(dW.(output_layer_gradient, layers[end-1])) + regularization * net.weights[end])

    grad = output_layer_gradient
    for i in (length(net.weights)-1):-1:1
        grad = [hid_grad(net.weights[i+1], g, l) for (g, l) in zip(grad, layers[i+1])]

        net.biases[i] -= α*mean(grad)
        net.weights[i] -= α*(mean(dW.(grad, layers[i])) + regularization*net.weights[i])
    end
end

function cost(net, x, t)
    o = forward(net, x)[end]
    -log(o[t])
end

function train(net::Net, x::Vector, t::Vector, epochs::Int, batch_size::Int, rate::Number, decay::Number = 1; regularization::Number = 0)
    plotly()

    indices = 1:length(x)
    test_indices = shuffle(indices)[1:2*batch_size]
    train_indices = filter(i -> !(i in test_indices), indices)

    mc = Vector{Float64}()
    test_accs = Vector{Float64}()
    train_accs = Vector{Float64}()

    for epoch in 1:epochs
        batch_indices = chop(shuffle(train_indices), batch_size)

        costs = nothing
        for bi in batch_indices
            update_weights(net, x[bi], t[bi], rate, regularization)
            costs = [cost(net, x[bi][j], t[bi][j]) for j in 1:length(bi)]
        end

        mean_cost = mean(costs)
        test_acc = accuracy(net, x[test_indices], t[test_indices])
        train_acc = accuracy(net, x[train_indices], t[train_indices])

        println("Epoch $epoch/$epochs")
        println("\tTrain accuracy: $(train_acc * 100) %")
        println("\tTest accuracy: $(test_acc * 100) %")
        println("\tMean cost: $mean_cost")
        println("\tLearning rate: $rate")
        
        push!(mc, mean_cost)
        push!(test_accs, test_acc)
        push!(train_accs, train_acc)

        rate /= decay
    end

    plt = plot(1:epochs, mc, xlabel = "Epoch", ylabel = "Value", label = "Mean cost", line = [:scatter], grid = false) # dpi = 600, size = (1440, 810)
    plot!(plt, 1:epochs, test_accs, line = [:scatter], label = "Test accuracy")
    plot!(plt, 1:epochs, train_accs, line = [:scatter], label = "Train accuracy")
    display(plt)
end

using Serialization
using Mmap
data = deserialize("trainset/data2")
targets = deserialize("trainset/targets2")

#data = [Float64.(Gray.(d)) for d in data]

N = 310
l = 900
#data = mmap("trainset/trainst3", BitMatrix, (l, N))
#targets = mmap("trainset/targets3", BitMatrix, (3, N))

#data = [data[:, i] for i in 1:N]
#targets = [argmax(targets[:, i]) for i in 1:N]

net = Net(l, 10, 3)
train(net, data, targets, 750, 50, 2e-2, regularization = 1e-2)