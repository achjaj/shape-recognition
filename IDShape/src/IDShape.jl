module IDShape
  include("neural.jl")
  
  using ArgParse
  using Images
  using ImageBinarization
  using Mmap

  function parseArgs()::Dict{Symbol, Any}
    parser = ArgParseSettings()

    @add_arg_table! parser begin
      "--train", "-t"
        help = "Training mode"
        arg_type = String
      "--weights", "w"
        help = "Path to saved weights"
        arg_type = String
        default = "weights"
      "data"
        help = "Eighter image to process or training data"
        required = true
    end

    parse_args(parser; as_symbols = true)
  end

  function loadMmaps(dataPath::String, targetsPath::String)
  end

  function trainingMode(net::Neural.Net, dataPath::String, targetsPath::String)
    data, targets = loadMmaps(dataPath, targetsPath)
    Neural.train!(net, data, targets, 100, 20, 1)
  end

  function julia_main()::Cint
    parsed = parseArgs()

    # now it creates neural net with:
    #   - input layer of output size 10 and with ReLu as activation function,
    #   - two hidden layers with sizes 5 and 10 and ReLu as activation functions
    #   - and output layer of size 3 with Softmax as activation function
    net = Neural.Net(10000, [10, 5, 10, 3], [:relu, :relu, :relu, :softmax])

    if typeof(parsed[:train]) == String
      trainingMode(net, parsed[:data], parsed[:train])

    else
      identify(net, parsed[:weights], parsed[:data])
    end

    return 0
  end

end # module

# test
IDShape.julia_main()