using Mmap
using ImageBinarization
using Images
using Colors
using Dates

log(msg::String) = println("$(Dates.now()) $msg")

dirs = ["/run/media/jakub/3315-3BCD/trainset3/circles", "/run/media/jakub/3315-3BCD/trainset3/triangles/", "/run/media/jakub/3315-3BCD/trainset3/rectangles/"] # input dirs

function toBitMatrix(img::Matrix)
    noalpha = RGB{N0f8}.(img)
    binarized = binarize(noalpha, Otsu())

    Bool.(binarized)
end

toVector(m::BitMatrix) = reshape(m, length(m), 1)[1:end]

function numberOfFiles(dir::String)
    output = readchomp(`src/numberOfFilesInDir.sh $dir`) # faster than readdir: does not collect file names

    return parse(Int64, output)
end


vectorLen = 30*30 # the images have these dimensions
totalNumberOfFiles = sum(numberOfFiles.(dirs))
println(totalNumberOfFiles)

dataFile = open("/home/jakub/Dokumenty/skola/siete/neural/trainset/trainset4", "w+") # output data file
dataMap = mmap(dataFile, BitMatrix, (vectorLen, totalNumberOfFiles))

targetsFile = open("/home/jakub/Dokumenty/skola/siete/neural/trainset/targets4", "w+") # output targets file
targetsMap = mmap(targetsFile, BitMatrix, (3, totalNumberOfFiles)) 

columnIndex = 1

for (i, dir) in enumerate(dirs)
    target = Bool.(==(i, j) for j in 1:3)

    log("Entering directory $dir")
    for file in readdir(dir, join = true)
        global columnIndex

        log("Loading file $file")
        img = load(file)
        log("Converting")
        img = toBitMatrix(img)
        img = toVector(img)

        log("Adding to matrix")
        dataMap[:, columnIndex] = img
        targetsMap[:, columnIndex] = target

        columnIndex += 1
    end
end