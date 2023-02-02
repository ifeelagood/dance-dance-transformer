#!/bin/bash
source vars.sh

mkdir -pv $RAW/fraxtil $RAW/itg

$PYSCRIPTS/download.py --path $RAW/fraxtil "https://fra.xtil.net/simfiles/data/tsunamix/III/Tsunamix III [SM5].zip" "https://fra.xtil.net/simfiles/data/arrowarrangements/Fraxtil's Arrow Arrangements [SM5].zip" "https://fra.xtil.net/simfiles/data/beastbeats/Fraxtil's Beast Beats [SM5].zip"
$PYSCRIPTS/download_smo.py --path $RAW/itg "In The Groove 1.zip" "In The Groove 2.zip"