include("neural.jl")
include("imgtransform.jl")
  
using ArgParse
using Mmap
using Images
using ImageTransformations
using Serialization # TODO: change to better format

names = ["circle", "triangle", "rectangle"]

function parseArgs()::Dict{Symbol, Any}
  parser = ArgParseSettings()

  @add_arg_table! parser begin
    "--train", "-t"
      help = "Training mode. This option takes a path to the targets matrix and the number of data points as arguments."
      metavar = ["TARGETS", "LENGTH"]
      nargs = 2
    "--weights", "-w"
      help = "Set path to the saved weights"
      arg_type = String
      default = "weights"
    "--test", "-T"
      help = "Test the trained neural network"
      metavar = ["TARGETS", "LENGTH"]
      nargs = 2
    "data"
      help = "Eighter image to process or a training data"
      required = true
  end

  parse_args(parser; as_symbols = true)
end

 # TODO: change to better format
function saveNet(net::Neural.Net, path::String)
  serialize(path, net.layers)
end

# TODO: change to better format
function loadNet!(net::Neural.Net, path::String)
  net.layers = deserialize(path)
end
  
function loadMmaps(dataPath::String, targetsPath::String, length::Int)
  dataMap = mmap(dataPath, BitMatrix, (900, length))
  targetsMap = mmap(targetsPath, BitMatrix, (3, length))

  return dataMap, targetsMap
end

function trainingMode(net::Neural.Net, dataPath::String, targetsPath::String, length::Int)
  data, targets = loadMmaps(dataPath, targetsPath, length)
  Neural.train!(net, data, targets, 100, 1000, 1e-2, 10/9)

  saveNet(net, "weights")
end

function testingMode(net::Neural.Net, weightsPath::String, dataPath::String, targetsPath::String, length::Int)
  data, targets = loadMmaps(dataPath, targetsPath, length)
  loadNet!(net, weightsPath)

  println("Testing accuracy is $(Neural.accuracy(net, data, targets) * 100)%")
end

# Preparse the image for the neural network
function loadAndPrepareImg(imgPath::String)
  img = load(imgPath)

  binarized = ImgTransform.imgToBitMatrix(img) # convert image to "negative" (swapped 0 and 1) BitMatrix
  cutout = ImgTransform.cutoutShape(binarized) # locate the shape on image, cut it out
  positive = xor.(cutout, 1) # change back to positive
  resized = imresize(positive, 30, 30) # resize; returns Matrix{Float64}
  resized = BitMatrix(round.(Int, resized))

  return ImgTransform.toVector(resized) # return vector
end

function identify(net::Neural.Net, weightsPath::String, imgPath::String)
  loadNet!(net, weightsPath)

  img = loadAndPrepareImg(imgPath)
  id, probs = Neural.identify(net, img)

  println("The shape in the image is a $(names[id]) ($(round(probs[id], digits=4) * 100)% confidence).")

  for (l, p) in zip(names, probs)
    print("$l: $p ")
  end
  println()
end


parsed = parseArgs()

# now it creates neural net with:
#   - hidden layer of output size 20 and with ReLu as activation function,
#   - and output layer of size 3 with Softmax as activation function
net = Neural.Net(900, [10, 3], [:relu, :softmax])

if length(parsed[:train]) > 0
  len = parse(Int, parsed[:train][2])
  trainingMode(net, parsed[:data], parsed[:train][1], len)
elseif length(parsed[:test]) > 0
  len = parse(Int, parsed[:test][2])
  testingMode(net, parsed[:weights], parsed[:data], parsed[:test][1], len)
else
  identify(net, parsed[:weights], parsed[:data])
end
