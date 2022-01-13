module Neural
    using Random
    using LinearAlgebra

    mutable struct Layer
        W::Matrix{Float64} # weights
        b::Vector{Float64} # bias

        afun::Function     # activaion function
        dafun::Function    # derivative of the activation function

        z::Vector{Float64} # W*x+b
        a::Vector{Float64} # afun(z)

        dCda::Vector{Float64}
        dCdW::Matrix{Float64}
        dCdb::Vector{Float64}

        function Layer(inlen::Int64, outlen::Int64, afun::Function, dafun::Function)
            σ = 2/(inlen + outlen)
            
            new(rand(outlen, inlen) * σ, rand(outlen) * σ, afun, dafun, [zeros(outlen) for i in 1:3]..., zeros(outlen, inlen), zeros(outlen))
        end
    end

    mutable struct Net
        inlen::Int64
        outlen::Int64
        layers::Vector{Layer}
        C::Function
        dC::Function

        function Net(inlen::Int64, outlen::Int64, sizes::Vector{Int64}, activations::Vector{Function}, dactivations::Vector{Function};
                     random_seed::Int64=42, C::Function = (o, t) -> sum((o - t).^2)/2, dC::Function = (o, t) -> o-t)

            Random.seed!(random_seed)
            
            layers = Vector{Layer}()
            ilen = inlen
            for (size, afun, dafun) in zip(sizes, activations[1:end-1], dactivations[1:end-1])
                push!(layers, Layer(ilen, size, afun, dafun))
                ilen = size
            end
            push!(layers, Layer(sizes[end], outlen, activations[end], dactivations[end]))

            new(inlen, outlen, layers, C, dC)
        end
    end

    @views function makechunks(X::AbstractVector, n::Integer)
        c = length(X) ÷ n
        return [X[1+c*k:(k == n-1 ? end : c*k+c)] for k = 0:n-1]
    end

    mean(x) = sum(x)/length(x)

    function forward!(net::Net, x::Vector)
        input = x

        for layer in net.layers
            layer.z = layer.W*input + layer.b
            layer.a = layer.afun(layer.z)

            input = layer.a
        end

        return input
    end

    function backward!(net::Net, x::Vector{Float64}, t::Vector{Float64})
       lastLayer = net.layers[end]

       lastLayer.dCda =  lastLayer.dafun(lastLayer.z) * net.dC(lastLayer.a, t)
       lastLayer.dCdW += lastLayer.dCda * transpose(net.layers[end - 1].a)
       lastLayer.dCdb += lastLayer.dCda

       for i in (length(net.layers) - 1):-1:1
            layer = net.layers[i]
            next = net.layers[i + 1]
            prevout = i == 1 ? x : net.layers[i - 1].a

            layer.dCda = layer.dafun(layer.a) * transpose(next.W) * next.dCda
            layer.dCdW += layer.dCda * transpose(prevout)
            layer.dCdb += layer.dCda
       end
    end

    function updateLayer!(layer::Layer, α::Float64)
        layer.W -= α*layer.dCdW
        layer.b -= α*layer.dCdb

        layer.dCda = zeros(size(layer.dCda))
        layer.dCdb = zeros(size(layer.dCdb))
        layer.dCdW = zeros(size(layer.dCdW))
    end

    predict(net::Net, x::Vector{Float64}) = argmax(forward!(net, x))

    function accuracy(net::Net, x::Vector{Vector{Float64}}, t::Vector{Vector{Float64}})
        predictions = map(i -> predict(net, i), x)
        cmp = [Int(predictions[i] == argmax(t[i])) for i in 1:length(t)]

        return sum(cmp)/length(t)
    end

    function train!(net::Net, x::Vector{Vector{Float64}}, t::Vector{Vector{Float64}}, batchSize::Int64, epochs::Int64, α::Float64, αDecay::Float64 = 1.0)
        for i in 1:epochs
            costs = Vector{Float64}()
            accs = Vector{Float64}()
            indices = shuffle(1:length(x))

            x = x[indices]
            t = t[indices]
            xBatches = makechunks(x, batchSize)
            tBatches = makechunks(t, batchSize)

            for (xBatch, tBatch) in zip(xBatches, tBatches)
                for (xi, ti) in zip(xBatch, tBatch)
                    output = forward!(net, xi)
                    push!(costs, net.C(output, ti))

                    backward!(net, xi, ti)
                end

                push!(accs, accuracy(net, x, t))
                updateLayer!.(net.layers, α)
            end
            
            α /= αDecay

            println("Epoch $i:")
            println("\tMean cost: $(mean(costs))")
            println("\tMean accuracy: $(mean(accs) * 100)")
        end
    end
end

using Main.Neural
using PyCall
using LinearAlgebra

atanh(x) = tanh.(x)
datanh(a) = [δ(i,j) * (1-a[i]^2) for i in 1:length(a), j in 1:length(a)]

function relu(z::AbstractArray)::AbstractArray
    return max.(0, z)
end

function drelu1(z::Number)::Number
    return z > 0 ? 1.0 : 0.0
end

function drelu(z::AbstractArray)::AbstractArray
    dvec = drelu1.(z)
    return diagm(dvec[:, 1])
end

function softmax(z::AbstractArray)::AbstractArray
    shift = maximum(z)

    z = exp.(z .- shift)
    total = sum(z)
    return z ./ total
end

function dsoftmax(z::AbstractArray)::AbstractArray
    n = length(z)

    eye = Array{Number}(I, n, n)
    w   = softmax(z)
    e   = ones(n, 1)

    return w * e' .* (eye - e * w')
end

function init()
    datasets = pyimport("sklearn.datasets")
    digits = datasets.load_digits()

    x = [digits["data"][i, :] for i in 1:size(digits["data"], 1)]
    t = map(i -> [δ(i, j) for j in 1:10], digits["target"] .+ 1)

    net = Neural.Net(length(digits["feature_names"]), 10, [20], [relu, softmax], [drelu, dsoftmax])
    Neural.train!(net, x, t, 100, 40, 1e-3)
end