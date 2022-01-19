# shape-recognition
Simple shape recognition by neural network.

Only triangles, rectangles and circles are recognized.

This program is part of my final exam.

## Trainset
The *trainset* directory contains two files: *data.7z* and *targets*.
The archive contains single file named *data*. Both *data* and *targets* files contains `mmap`-ed `BitMatrix`.
The *data* matrix has size 10000x111907 and contains the training images. Every image was binarized,
converted to `BitMatrix` and then converted to column `BitVector`. (see [imgsToSingleData.jl](imgsToSingleData.jl))

The *targets* matrix has size 3x111907 and contains the coresponding targets. Every columnt vector
has one non-zero element which is at the position of the category index.
The categories have following indices:

- Circle: 1
- rectangle: 2
- Triangle: 3
