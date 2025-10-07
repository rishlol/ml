#!/bin/bash

BUILD_SYSTEM=""
if [ $# -ge 1 ]; then
	BUILD_SYSTEM="-G $1"
	echo "Using $1 build system..."
else
	echo "Using default build system..."
fi

cmake -S . -B build $BUILD_SYSTEM
