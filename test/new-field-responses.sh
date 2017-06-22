#!/bin/bash
#
# Quick and dirty processing of new field responses
#

usage () {
    new-field-responses /path/to/ub_10.tar.gz
}

speed="1.114*mm/us"
origin="10*cm"

infile=$1 ; shift
base=$(echo $(basename $infile .tar.gz) | tr '_' '-')
destdir=$(dirname $infile)

do_one () {
    local type=$1 ; shift
    local norm=0
    if [ $type = "wnormed" ] ; then
        norm=-1
    fi
    

    local stub="${destdir}/${base}-${type}"
    set -x
    wirecell-sigproc convert-garfield -n $norm -s $speed -o $origin $infile "${stub}.json.bz2"

    wirecell-sigproc plot-garfield-exhaustive     -n $norm       $infile "${stub}-exhaustive.pdf"

    wirecell-sigproc plot-garfield-track-response -n $norm -s 0  $infile "${stub}-current.pdf"
    wirecell-sigproc plot-garfield-track-response -n $norm -a 0  $infile "${stub}-voltage.pdf"
    wirecell-sigproc plot-garfield-track-response -n $norm       $infile "${stub}-adc.pdf"
    set +x
}

do_one absolute
do_one wnormed
