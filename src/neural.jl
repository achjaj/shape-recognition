module Neural
    using Random
    include("functions.jl")

    mutable struct Layer
        W::AbstractArray{Float64} # weights
        b::Vector{Float64} # bias

        afun::Function     # activaion function
        dafun::Function    # derivative of the activation function

        z::Vector{Float64} # W*x+b
        a::Vector{Float64} # afun(z)

        dCda::Vector{Float64}
        dCdW::AbstractArray{Float64}
        dCdb::Vector{Float64}

        function Layer(inlen::Int64, outlen::Int64, afun::Function, dafun::Function)
            σ = 2/(inlen + outlen)
            
            new(rand(outlen, inlen) * σ, rand(outlen) * σ, afun, dafun, [zeros(outlen) for i in 1:3]..., zeros(outlen, inlen), zeros(outlen))
        end
    end

    mutable struct Net
        layers::Vector{Layer}
        C::Function
        dC::Function

        function Net(inlength::Int64, sizes::Vector{Int64}, activations::Vector; # activations can be eighter vector of symbols or vector of 
                                                                                 # Tuple{Function, Function} where the 2nd function is derivative of the 1st
                    C::Function = leasSquareCost, dC::Function = dLeastSquareCost)

            layers = Vector{Layer}()
            ilen = inlength
            for (size, id) in zip(sizes, activations)
                push!(layers, Layer(ilen, size, getFunctions(id)...))
                ilen = size
            end

            new(layers, C, dC)
        end
    end

    @views function makechunks(X::AbstractArray, n::Integer)
        c = length(X) ÷ n
        return [X[1+c*k:(k == n-1 ? end : c*k+c)] for k = 0:n-1]
    end

    mean(x) = sum(x)/length(x)

    leasSquareCost(o, t) = sum((o - t).^2)/2
    dLeastSquareCost(o, t) = o-t

    getFunctions(id::Symbol) = activations[id]

    getFunctions(f::Tuple{Function, Function}) = f

    function applyLayer!(layer::Layer, x::AbstractArray)
        layer.z = layer.W*x + layer.b
        layer.a = layer.afun(layer.z)

        return layer.a
    end

    function forward!(net::Net, x::AbstractArray)
        input = x
        map(layer -> input = applyLayer!(layer, input), net.layers)

        return input
    end

    function backward!(net::Net, x::AbstractArray, t::AbstractArray)
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

    function identify(net::Net, x::AbstractArray)
        vec = forward!(net, x)
        id = argmax(vec)

        return id, vec
    end

    function accuracy(net::Net, x::AbstractArray, t::AbstractArray)
        identities = map(i -> identify(net, x[:, i])[1], 1:size(x, 2))
        cmp = [Int(identities[i] == argmax(t[:, i])) for i in 1:size(t, 2)]

        return sum(cmp)/size(t, 2)
    end

    function train!(net::Net, x::AbstractArray, t::AbstractArray, batchSize::Int64, epochs::Int64, α::Float64, αDecay::Float64 = 1.0, epochSaveF = nothing, increaseAt::Int = 3)
    lastCost = 1e5
    successiveDecrease = 0

    for i in 1:epochs
        costs = Vector{Float64}()
        accs = Vector{Float64}()
        indices = shuffle(1:size(x, 2)) # one column represents one data point

        batches = makechunks(indices, batchSize)

        for batch in batches
            for i in batch
                output = forward!(net, x[:, i])
                push!(costs, net.C(output, t[:, i]))

                backward!(net, x[:, i], t[:, i])
            end

            push!(accs, accuracy(net, x, t))
            updateLayer!.(net.layers, α)
        end

        meanCost = mean(costs)
        meanAcc = mean(accs) * 100
        println("Epoch $i/$epochs:")
        println("\tMean cost: $meanCost")
        println("\tMean accuracy: $meanAcc%")
        println("\tLearning rate: $α")

        if epochSaveF !== nothing
            epochSaveF(net, "epochs/e$(i)_$(round(Int, meanAcc))")
        end

        if meanCost < lastCost
            successiveDecrease += 1
        elseif meanCost > lastCost
            α /= αDecay
            successiveDecrease = 0
        end

        if successiveDecrease == increaseAt
            α *= αDecay
            successiveDecrease = 0
        end

        lastCost = meanCost
    end
end

    
end
