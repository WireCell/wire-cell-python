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

wirecell-sigproc convert-garfield -s $speed -o $origin $infile "${destdir}/${base}.json.bz2"

wirecell-sigproc plot-garfield-track-response -s 0  $infile "${destdir}/${base}-current.pdf"
wirecell-sigproc plot-garfield-track-response -a 0  $infile "${destdir}/${base}-voltage.pdf"
wirecell-sigproc plot-garfield-track-response       $infile "${destdir}/${base}-adc.pdf"
