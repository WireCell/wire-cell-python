#!/bin/bash

uboone_tarball=/opt/bviren/wct-dev/share/wirecell/data/ub_10.tar.gz
dune_tarball=/opt/bviren/wct-dev/share/wirecell/data/dune_4.71.tar.gz

set -x
wirecell-sigproc plot-garfield-track-response \
                 --ymin -60 \
                 --ymax 100 \
                 --electrons 20881 \
                 --adc-gain 1.0 \
                 --adc-voltage 1.4 \
                 --tick-padding 40 \
                 $dune_tarball track-response-dune.pdf

wirecell-sigproc plot-garfield-track-response \
                 $uboone_tarball track-response-uboone.pdf
