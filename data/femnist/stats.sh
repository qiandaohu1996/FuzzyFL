#!/usr/bin/env bash

NAME="femnist"

cd ../util

python stats.py --name $NAME

cd ../$NAME