# shape-recognition
Simple shape recognition by a neural network.

Only triangles, rectangles and circles are recognized.

This program is part of my final exam.

## Trainset
The *trainset* directory contains two files: *data.7z* and *targets*.
The archive contains a single file named *data*. Both *data* and *targets* files contain `mmap`-ed `BitMatrix`.

The *data* matrix has a size 10000x111907 and contains the training images. Every image was binarized,
converted to `BitMatrix` and then converted to column `BitVector`. (see [imgsToSingleData.jl](imgsToSingleData.jl))

The *targets* matrix has a size 3x111907 and contains the coresponding targets. Every column vector
has one non-zero element which is at the position of the category index.
The categories have the following indices:

- Circle: 1
- Triangle: 2
- Rectangle: 3

## Dependencies
 - [Julia](https://julialang.org/)
 - [ArgParse.jl](https://github.com/carlobaldassi/ArgParse.jl)
 - [Images.jl](https://github.com/JuliaImages/Images.jl)
 - [ImageTransformations.jl](https://github.com/JuliaImages/ImageTransformations.jl)
 - [ImageBinarization.jl](https://github.com/zygmuntszpak/ImageBinarization.jl)

## Usage
```
usage: julia IDShape.jl [-t TRAIN] [-w WEIGHTS] [-h] data

positional arguments:
  data                  Eighter image to process or training data

optional arguments:
  -t, --train TRAIN     Training mode
  -w, --weights WEIGHTS
                        Path to saved weights (default: "weights")
  -h, --help            show this help message and exit
```
The main program file is [IDShape.jl](src/IDShape.jl) inside the *src* folder.

The input image can be in any format supported by [Images](https://github.com/JuliaImages/Images.jl) (PNG, JPEG, etc.) and it can have arbitrary dimensions.

The NN is already trained and the weights are saved in *weights* file in the root folder.