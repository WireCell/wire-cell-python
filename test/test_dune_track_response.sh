#!/bin/bash

garfield_tarball=/opt/bviren/wct-dev/share/wirecell/data/dune_4.71.tar.gz

set -x
wirecell-sigproc plot-garfield-track-response \
                 --ymin -60 \
                 --ymax 100 \
                 --electrons 20881 \
                 --adc-gain 1.0 \
                 --adc-voltage 1.4 \
                 $garfield_tarball dune-track.pdf
