δ(i, j) = Int64(==(i, j))

diagm(x::Vector) = [δ(i, j)*x[i] for i in 1:length(x), j in 1:length(x)]

I(n) = [δ(i, j) for i = 1:n, j = 1:n]

vtanh(x::Vector) = tanh.(x)

dtanh(x::Vector) = diagm([i^2 for i in x])

relu(z::Vector) = max.(0, z)

leasSquareCost(o, t) = sum((o - t).^2)/2
dLeastSquareCost(o, t) = o-t

softmaxLikelihood(o, t) = -log(o[argmax(t)])
dSoftmaxLikelihood(o, t) = o - t

function drelu(z::Vector)
  dvec = Int64.(>(0).(z))
  return diagm(dvec[:, 1])
end

function softmax(z::Vector)
  shift = maximum(z)

  z = exp.(z .- shift)
  total = sum(z)
  return z ./ total
end

function dsoftmax(z::Vector)
  n = length(z)

  w = softmax(z)
  e = ones(n, 1)

  return w * e' .* (I(n) - e * w')
end

mean(x) = sum(x)/length(x)

activations = Dict(:vtanh => (vtanh, dtanh), :relu => (relu, drelu), :softmax => (softmax, dsoftmax))