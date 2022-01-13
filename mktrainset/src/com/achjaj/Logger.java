package com.achjaj;

public class Logger {

    static void log(String msg) {
        System.out.println(msg);
    }

    static void log(String... msg) {
        log(String.join(" ", msg));
    }
}
