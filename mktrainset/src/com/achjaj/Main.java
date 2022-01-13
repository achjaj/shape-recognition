package com.achjaj;

import java.io.File;
import java.io.IOException;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.stream.Stream;

public class Main {

    public static void main(String[] args) throws IOException {
        if (args.length == 0)
            printUsageEndExit();

        var processor = new Processor();
        var inputFiles = Arrays.copyOfRange(args, 1, args.length/2 + 1);
        var output = Arrays.copyOfRange(args, args.length/2 + 1, args.length);

        for (int index = 0; index < inputFiles.length; index++) {
            Logger.log("Processing image:", inputFiles[index], "->", output[index], "; Action:", args[0]);

            switch (args[0]) {
                case "split" -> {
                    Logger.log("Creating directory (recursively):", output[index]);
                    try {
                        Files.createDirectories(Path.of(output[index]));
                    } catch (FileAlreadyExistsException e) {/*IGNORE*/}

                    processor.split(new File(inputFiles[index]), output[index], 20);
                }
                case "mark" -> processor.mark(new File(inputFiles[index]), output[index], 20);
            }
        }

    }

    static void printUsageEndExit() {
        System.out.println("Usage: mktrainset <action> <input files> <output directories>");
        System.out.println("\tActions:\n\t\tsplit\n\t\ttransform\n\t\tmark");
        System.out.println("\nNUMBER OF INPUT FILES MUST BE EQUAL TO NUMBER OF OUTPUT DIRECTORIES!");
        System.exit(1);
    }
}
