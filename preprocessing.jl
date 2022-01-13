using ImageTransformations
using ImageBinarization
using Images

struct Rectangle
    xl::Int64 # x coordinate of top left corner
    yl::Int64 # y ccordinate of top left corner
    xr::Int64 # x coordinate of bottom right corner
    yr::Int64 # y coordinate of bottom right corner
end

function escribedRectangle(binarized)
    filter = p -> p ≠ 0 # filter used to find relevant points 

    h, w = size(binarized)
    rows = [sum(binarized[i, :]) for i in 1:h] # map binarized to the vector of sums of rows, i.e. rows[i] = sum of elements in the i-th row
    columns = [sum(binarized[:, i]) for i in 1:w] # same as before but for columns

    # find relevant points/lines
    top = findfirst(filter, rows) # get index of first non-zero element
    bottom = findlast(filter, rows)
    left = findfirst(filter, columns)
    right = findlast(filter, columns)

    Rectangle(left, top, right, bottom)
end

function process(img, size::Tuple{Int64, Int64})
    binarized = binarize(img, Otsu()) # binarize using Otsu algorithm; returns Matrix{Gray}
    binarized = Bool.(binarized) # convert Matrix{Gray} to BitMatrix
    binarized = xor.(binarized, 1) # make "negative image"; binarize function transform white color to 1 and black to 0, I want it the other way around

    rect = escribedRectangle(binarized)
    cut = binarized[rect.yl:rect.yr, rect.xl:rect.xr] # cut out the rectangle containing the drawing

    imresize(cut, size)
end


testimage = load("/home/jakub/Dokumenty/skola/siete/neural/trainset/triangles/84.bmp")
processed = process(testimage, (500, 500))
