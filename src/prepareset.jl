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
        image = xor.(image, 1)

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

inputs = ["/home/jakub/Dokumenty/skola/siete/neural/presets/circles/", "/home/jakub/Dokumenty/skola/siete/neural/presets/rectangles/", "/home/jakub/Dokumenty/skola/siete/neural/presets/triangles/"] # input directories
outputs = ["/run/media/jakub/3315-3BCD/trainset3/circles", "/run/media/jakub/3315-3BCD/trainset3/rectangles", "/run/media/jakub/3315-3BCD/trainset3/triangles"] # output directories


rotations = [x -> imrotate(x, θ, fillvalue=1) for θ in (10*π/180):(10*π/180):2π]
resize = [i -> imresize(i, 30, 30)]
rotresize = map(rot -> x -> imresize(rot(x), 30, 30), rotations)
#TODO: add more transfomations; gaussian noise, add blob to random place, remove random part of the shape

transformations = [rotresize]
#transformations = [rotations]
#transformations = [[x -> imrotate(x, π/4, fillvalue=1)]]

log("Creating output directories")
mkpath.(outputs)

for (input, output) in zip(inputs, outputs)
    log("Entering directory: $input")
    for (i, transforms) in enumerate(transformations)
        log("\tUsing transformations set: $i")
        processInput(input, transforms, output)
    end
end

