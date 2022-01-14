include("imgtransform.jl")
using .ImgTransform
using ImageTransformations
using Images
using DelimitedFiles
using Dates
using Colors

randStr(len::Int64) = String(UInt8.(rand([97:122; 65:90; 48:57], len)))

log(msg::String) = println("$(Dates.now()) $msg")

function processInput(inputDir::String, transformations, outputDir::String)
    paths = joinpath.(inputDir, readdir(inputDir))

    for path in paths
        log("Loading image: $path")
        image = load(path)
        log("Binarization")
        image = ImgTransform.imgToBitMatrix(image)
        log("Cutting out shape")
        image = ImgTransform.cutoutShape(image)

        for (i, transform) in enumerate(transformations)
            log("Selecting $i-th transformation")
           # log("Looking for available random name")
            newpath = joinpath(outputDir, randStr(rand(5:10))*".bmp")
            #while isfile(path)
            #    newpath = joinpath(outputDir, randStr(rand(1:15)))
            #end

            log("Transforming")
            transformed = transform(image)
            result = Gray.(transformed)

            log("Writing transformed image to: $newpath")
            save(newpath, result)
        end
    end
end

inputs = ["presets/triangles/", "presets/rectangles/", "presets/circles/"]
outputs = ["trainsets/triangles", "trainsets/rectangles", "trainsets/circles"]

rotations = [x -> imrotate(x, θ) for θ in (π/180):(π/180):2π]
resize = [i -> imresize(i, 100, 100)]
#TODO: add more transfomations; gaussian noise, add blob to random place, remove random part of the shape

transformations = [resize]

log("Creating output directories")
mkpath.(outputs)

for (input, output) in zip(inputs, outputs)
    log("Entering directory: $input")
    for (i, transforms) in enumerate(transformations)
        log("\tUsing transformations set: $i")
        processInput(input, transforms, output)
    end
end

