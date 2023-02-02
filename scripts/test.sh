#!/bin/bash


echo "Removing old data..."
rm -rf ../data

echo "Setting up directories..."
./00-setup.sh

echo "Downloading files..."
./01-download.sh

echo "Unpacking files..."
./02-extract.sh