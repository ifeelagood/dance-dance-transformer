#!/bin/bash

if [ $(basename $(pwd)) == "scripts" ]; then
    cd ..
fi

SCRIPTS=$(pwd)/scripts
PYSCRIPTS=$(pwd)/pyscripts
DATA=$(pwd)/data

RAW=$DATA/raw