package com.achjaj;

import boofcv.abst.filter.binary.BinaryLabelContourFinder;
import boofcv.abst.filter.binary.InputToBinary;
import boofcv.abst.shapes.polyline.ConfigPolylineSplitMerge;
import boofcv.abst.shapes.polyline.PointsToPolyline;
import boofcv.alg.filter.binary.BinaryImageOps;
import boofcv.alg.filter.binary.Contour;
import boofcv.factory.filter.binary.ConfigThreshold;
import boofcv.factory.filter.binary.FactoryBinaryContourFinder;
import boofcv.factory.filter.binary.FactoryThresholdBinary;
import boofcv.factory.filter.binary.ThresholdType;
import boofcv.factory.shape.FactoryPointsToPolyline;
import boofcv.io.image.ConvertBufferedImage;
import boofcv.struct.ConfigLength;
import boofcv.struct.image.GrayF32;
import boofcv.struct.image.GrayS32;
import boofcv.struct.image.GrayU8;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Comparator;
import java.util.List;

public class Processor {
    private BinaryLabelContourFinder binaryToContour;
    private InputToBinary<GrayF32> inputToBinary;

    public Processor() {
        var config = new ConfigPolylineSplitMerge();
        config.minimumSides = 3;
        config.maximumSides = 7;
        config.convex = true;
        config.loops = true;
        config.maxSideError = new ConfigLength(3, 0.05);
        config.minimumSideLength = 4;
        config.cornerScorePenalty = 0.125;

        var config2 = new ConfigThreshold();
        config2.type = ThresholdType.GLOBAL_OTSU;
        config2.down = true;
        inputToBinary = FactoryThresholdBinary.threshold(config2, GrayF32.class);

        binaryToContour = FactoryBinaryContourFinder.linearChang2004();
    }

    List<Contour> locate(BufferedImage image) {
        Logger.log("Looking for polygons");
        var input = ConvertBufferedImage.convertFromSingle(image, null, GrayF32.class);
        var binary = new GrayU8(input.width, input.height);
        var labeled = new GrayS32(1, 1);

        inputToBinary.process(input, binary);
        binaryToContour.process(binary, labeled);

        var result = BinaryImageOps.convertContours(binaryToContour);
        Logger.log("Found", result.size() + "", "polygons");

        return result;
    }

    Rectangle escribedRectangle(Contour contour, int buffer) {
        var points = contour.external;

        var top = points.stream().min(Comparator.comparingDouble(a -> a.y)).get().y - buffer;
        var bottom = points.stream().max(Comparator.comparingDouble(a -> a.y)).get().y + buffer;
        var left = points.stream().min(Comparator.comparingDouble(a -> a.x)).get().x - buffer;
        var right = points.stream().max(Comparator.comparingDouble(a -> a.x)).get().x + buffer;

        var height = bottom - top;
        var width = right - left;

        return new Rectangle(left, top, width, height);
    }

    Rectangle normalizeRectangle(Rectangle rectangle, int imgWidth, int imgHeight) {
        if (rectangle.x < 0)
            rectangle.x = 0;

        if (rectangle.y < 0)
            rectangle.y = 0;

        if (rectangle.x + rectangle.width > imgWidth)
            rectangle.width = imgWidth - rectangle.x;

        if (rectangle.y + rectangle.height > imgHeight)
            rectangle.height = imgHeight - rectangle.y;

        return rectangle;
    }

    void split(File input, String outDir, int buffer) throws IOException {
        var img = ImageIO.read(input);
        int[] counter = new int[]{0};

        locate(img).stream().map(c -> escribedRectangle(c, buffer))
            .map(rectangle -> normalizeRectangle(rectangle, img.getWidth(), img.getHeight()))
            .forEach(rectangle -> {
                var cut = new BufferedImage(rectangle.width, rectangle.height, BufferedImage.TYPE_BYTE_BINARY);
                var data = img.getRGB(rectangle.x, rectangle.y, rectangle.width, rectangle.height, null, 0, rectangle.width);
                cut.setRGB(0, 0, rectangle.width, rectangle.height, data, 0, rectangle.width);

                try {
                    Logger.log("Saving cutout", counter[0] + "");
                    ImageIO.write(cut, "BMP", new File(outDir, counter[0]++ + ".bmp"));
                } catch (IOException e) {
                    e.printStackTrace();
                }
            });
    }

    void mark(File input, String output, int buffer) throws IOException {
        var img = ImageIO.read(input);

        var outImg = new BufferedImage(img.getWidth(), img.getHeight(), img.getType());
        outImg.setData(img.getData());

        var pen = outImg.createGraphics();
        pen.setStroke(new BasicStroke(9));
        pen.setColor(Color.RED);

        locate(img).stream().map(c -> escribedRectangle(c, buffer)).forEach(pen::draw);

        ImageIO.write(outImg, "PNG", new File(output));
    }
}
