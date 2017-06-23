#!/bin/bash
#
# Quick and dirty processing of new field responses.  This is bv-specific.
#

usage () {
    new-field-responses 
}

speed="1.114*mm/us"
origin="10*cm"

do_one () {
    local infile=$1 ; shift
    local type=$1 ; shift
    local norm=0
    if [ $type = "wnormed" ] ; then
        norm=-1                 # by average collection integral
    fi
    if [ $type = "half" ] ; then
        norm=0.5                # roughly fix Garfield's factor of ~2
    fi

    local base=$(echo $(basename $infile .tar.gz) | tr '_' '-')
    local destdir=$(dirname $infile)


    local datastub="${datadir}/${base}-${type}"
    local plotstub="${datadir}/../plots/${base}-${type}"

    echo "NEW $datastub"

    set -x
    wirecell-sigproc convert-garfield -n $norm -s $speed -o $origin $infile "${datastub}.json.bz2"

    wirecell-sigproc plot-garfield-exhaustive     -n $norm       $infile "${plotstub}-exhaustive.pdf"

    wirecell-sigproc plot-garfield-track-response -n $norm -s 0  $infile "${plotstub}-current.pdf"
    wirecell-sigproc plot-garfield-track-response -n $norm -a 0  $infile "${plotstub}-voltage.pdf"
    wirecell-sigproc plot-garfield-track-response -n $norm       $infile "${plotstub}-adc.pdf"
    set +x

    echo
    
}



datadir=/opt/bviren/wct-dev/share/wirecell/data
for sample in ub_10 ub_10_uv_ground ub_10_vy_ground
do
    file="${sample}.tar.gz"
    for thing in absolute wnormed half
    do
        log="$sample-$thing.log"
        echo $log
        do_one "${datadir}/${file}" $thing > $log 2>&1 &
    done
done

