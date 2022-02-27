include("neural2.jl")
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
function saveNet(net::Net, path::String)
  serialize(path, net)
end

# TODO: change to better format
loadNet(path::String) = deserialize(path)
  
function loadMmaps(dataPath::String, targetsPath::String, length::Int)
  dataMap = mmap(dataPath, BitMatrix, (900, length))
  targetsMap = mmap(targetsPath, BitMatrix, (3, length))

  return dataMap, targetsMap
end

function trainingMode(net::Neural.Net, dataPath::String, targetsPath::String, length::Int)
  data, targets = loadMmaps(dataPath, targetsPath, length)
  train!(net, data, targets, 150, 100, 1e-2, 10/9)

  saveNet(net, "weights")
end

function testingMode(weightsPath::String, dataPath::String, targetsPath::String, length::Int)
  data, targets = loadMmaps(dataPath, targetsPath, length)
  net = loadNet!(weightsPath)

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

function identify(weightsPath::String, imgPath::String)
  net = loadNet(weightsPath)

  img = loadAndPrepareImg(imgPath)
  probs = forward(net, img)[end]
  id = argmax(probs)

  println("The shape in the image is a $(names[id]) ($(round(probs[id], digits=4) * 100)% confidence).")

  for (l, p) in zip(names, probs)
    print("$l: $p ")
  end
  println()
end


parsed = parseArgs()

if length(parsed[:train]) > 0
  len = parse(Int, parsed[:train][2])
  trainingMode(net, parsed[:data], parsed[:train][1], len)
elseif length(parsed[:test]) > 0
  len = parse(Int, parsed[:test][2])
  testingMode(parsed[:weights], parsed[:data], parsed[:test][1], len)
else
  identify(parsed[:weights], parsed[:data])
end
