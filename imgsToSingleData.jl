using Mmap
using ImageBinarization
using Images
using Colors
using Dates

log(msg::String) = println("$(Dates.now()) $msg")

dirs = ["/run/media/jakub/3315-3BCD/trainset/circles", "/run/media/jakub/3315-3BCD/trainset/triangles/", "/run/media/jakub/3315-3BCD/trainset/rectangles/"]

function toBitMatrix(img::Matrix)
    noalpha = RGB{N0f8}.(img)
    binarized = binarize(noalpha, Otsu())

    Bool.(binarized)
end

toVector(m::BitMatrix) = reshape(m, length(m), 1)[1:end]

function numberOfFiles(dir::String)
    output = readchomp(`./numberOfFilesInDir.sh $dir`) # faster than readdir: does not collect file names

    return parse(Int64, output)
end


vectorLen = 100*100 # the images have these dimensions
totalNumberOfFiles = sum(numberOfFiles.(dirs))

dataFile = open("/run/media/jakub/3315-3BCD/trainset/data", "w+")
dataMap = mmap(dataFile, BitMatrix, (vectorLen, totalNumberOfFiles))

targetsFile = open("/run/media/jakub/3315-3BCD/trainset/targets", "w+")
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