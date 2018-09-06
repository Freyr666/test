#!/bin/sh

CC=gcc
FLAGS=$( pkg-config --libs --cflags MagickWand glu egl gbm)

$CC $FLAGS -lm main.c libbmp.c -o test
