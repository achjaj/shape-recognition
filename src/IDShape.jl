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
      help = "Training mode"
      arg_type = String
    "--weights", "-w"
      help = "Path to saved weights"
      arg_type = String
      default = "weights"
    "data"
      help = "Eighter image to process or training data"
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
  
function loadMmaps(dataPath::String, targetsPath::String)
  dataMap = mmap(dataPath, BitMatrix, (10000, 111907))
  targetsMap = mmap(targetsPath, BitMatrix, (3, 111907))

  indices = [[rand(1:54508) for i in 1:700]..., [rand(54509:86638) for i in 1:700]..., [rand(86639:111907) for i in 1:700]...]

  return dataMap[:, indices], targetsMap[:, indices]
end

function trainingMode(net::Neural.Net, dataPath::String, targetsPath::String)
  data, targets = loadMmaps(dataPath, targetsPath)
  Neural.train!(net, data, targets, 100, 100, 1e-3, 3/4, saveNet)

  saveNet(net, "weights")
end

function loadAndPrepareImg(imgPath::String)
  img = load(imgPath)

  binarized = ImgTransform.imgToBitMatrix(img) # convert image to "negative" (swapped 0 and 1) BitMatrix
  cutout = ImgTransform.cutoutShape(binarized) # locate the shape on image, cut it out
  positive = xor.(cutout, 1) # change back to positive
  resized = imresize(positive, 100, 100) # resize; returns Matrix{Float64}
  resized = BitMatrix(round.(Int, resized))

  return ImgTransform.toVector(resized) # return vector
end

function identify(net::Neural.Net, weightsPath::String, imgPath::String)
  loadNet!(net, weightsPath)

  img = loadAndPrepareImg(imgPath)
  id, probs = Neural.identify(net, img)

  println("The shape on the image is a $(names[id]) ($(round(probs[id], digits=4) * 100)% confidence).")
end


parsed = parseArgs()

# now it creates neural net with:
#   - hidden layer of output size 20 and with ReLu as activation function,
#   - and output layer of size 3 with Softmax as activation function
net = Neural.Net(10000, [20, 3], [:relu, :softmax])


if typeof(parsed[:train]) == String
  trainingMode(net, parsed[:data], parsed[:train])

else
  identify(net, parsed[:weights], parsed[:data])
end
