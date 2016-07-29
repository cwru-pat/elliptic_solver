#!/bin/bash

# Just try to compile and run for now.
g++ main.cpp multigrid.cpp -O3 -Wall --std=c++11 -fopenmp
if [ $? -ne 0 ]; then
    echo "Error: compile failed."
    exit 1
fi

time ./a.out
if [ $? -ne 0 ]; then
    echo "Error: run failed."
    exit 1
fi
