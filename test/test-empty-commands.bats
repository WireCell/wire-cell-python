#!/usr/bin/env bats

# This file is generated.  Edits may be lost.
# See: /home/bv/wrk/wct/check-test/python/test/gen-test-empty-commands.sh

@test "assure wirecell-aux plot-idft-bench handles empty call" {
      local got="$( wirecell-aux plot-idft-bench 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-aux run-idft handles empty call" {
      local got="$( wirecell-aux run-idft 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-aux run-idft-bench handles empty call" {
      local got="$( wirecell-aux run-idft-bench 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-gen depo-lines handles empty call" {
      local got="$( wirecell-gen depo-lines 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-gen depo-point handles empty call" {
      local got="$( wirecell-gen depo-point 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-gen depo-sphere handles empty call" {
      local got="$( wirecell-gen depo-sphere 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-gen frame-stats handles empty call" {
      local got="$( wirecell-gen frame-stats 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-gen move-depos handles empty call" {
      local got="$( wirecell-gen move-depos 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-gen plot-depos handles empty call" {
      local got="$( wirecell-gen plot-depos 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-gen plot-sim handles empty call" {
      local got="$( wirecell-gen plot-sim 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-gen plot-test-boundaries handles empty call" {
      local got="$( wirecell-gen plot-test-boundaries 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-gen unitify-depos handles empty call" {
      local got="$( wirecell-gen unitify-depos 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-img activity handles empty call" {
      local got="$( wirecell-img activity 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-img anidfg handles empty call" {
      local got="$( wirecell-img anidfg 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-img bee-blobs handles empty call" {
      local got="$( wirecell-img bee-blobs 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-img blob-activity-mask handles empty call" {
      local got="$( wirecell-img blob-activity-mask 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-img dump-bb-clusters handles empty call" {
      local got="$( wirecell-img dump-bb-clusters 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-img dump-blobs handles empty call" {
      local got="$( wirecell-img dump-blobs 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-img inspect handles empty call" {
      local got="$( wirecell-img inspect 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-img paraview-activity handles empty call" {
      local got="$( wirecell-img paraview-activity 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-img paraview-blobs handles empty call" {
      local got="$( wirecell-img paraview-blobs 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-img paraview-depos handles empty call" {
      local got="$( wirecell-img paraview-depos 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-img plot-blobs handles empty call" {
      local got="$( wirecell-img plot-blobs 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-img plot-depos-blobs handles empty call" {
      local got="$( wirecell-img plot-depos-blobs 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-img transform-depos handles empty call" {
      local got="$( wirecell-img transform-depos 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-img wire-slice-activity handles empty call" {
      local got="$( wirecell-img wire-slice-activity 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-pgraph dotify handles empty call" {
      local got="$( wirecell-pgraph dotify 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-plot channel-correlation handles empty call" {
      local got="$( wirecell-plot channel-correlation 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-plot comp1d handles empty call" {
      local got="$( wirecell-plot comp1d 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-plot digitizer handles empty call" {
      local got="$( wirecell-plot digitizer 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-plot frame handles empty call" {
      local got="$( wirecell-plot frame 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-plot frame-diff handles empty call" {
      local got="$( wirecell-plot frame-diff 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-plot frame-image handles empty call" {
      local got="$( wirecell-plot frame-image 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-plot frame-means handles empty call" {
      local got="$( wirecell-plot frame-means 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-plot ntier-frames handles empty call" {
      local got="$( wirecell-plot ntier-frames 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-pytorch make-dft handles empty call" {
      local got="$( wirecell-pytorch make-dft 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-resp gf-info handles empty call" {
      local got="$( wirecell-resp gf-info 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-resp gf2npz handles empty call" {
      local got="$( wirecell-resp gf2npz 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-sigproc channel-responses handles empty call" {
      local got="$( wirecell-sigproc channel-responses 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-sigproc convert-electronics-response handles empty call" {
      local got="$( wirecell-sigproc convert-electronics-response 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-sigproc convert-garfield handles empty call" {
      local got="$( wirecell-sigproc convert-garfield 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-sigproc convert-noise-spectra handles empty call" {
      local got="$( wirecell-sigproc convert-noise-spectra 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-sigproc fr2npz handles empty call" {
      local got="$( wirecell-sigproc fr2npz 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-sigproc frzero handles empty call" {
      local got="$( wirecell-sigproc frzero 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-sigproc plot-configured-spectra handles empty call" {
      local got="$( wirecell-sigproc plot-configured-spectra 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-sigproc plot-electronics-response handles empty call" {
      local got="$( wirecell-sigproc plot-electronics-response 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-sigproc plot-garfield-exhaustive handles empty call" {
      local got="$( wirecell-sigproc plot-garfield-exhaustive 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-sigproc plot-garfield-track-response handles empty call" {
      local got="$( wirecell-sigproc plot-garfield-track-response 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-sigproc plot-noise-spectra handles empty call" {
      local got="$( wirecell-sigproc plot-noise-spectra 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-sigproc plot-response handles empty call" {
      local got="$( wirecell-sigproc plot-response 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-sigproc plot-response-compare-waveforms handles empty call" {
      local got="$( wirecell-sigproc plot-response-compare-waveforms 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-sigproc Plot handles empty call" {
      local got="$( wirecell-sigproc Plot 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-sigproc plot-response-conductors handles empty call" {
      local got="$( wirecell-sigproc plot-response-conductors 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-sigproc plot-spectra handles empty call" {
      local got="$( wirecell-sigproc plot-spectra 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-sigproc response-info handles empty call" {
      local got="$( wirecell-sigproc response-info 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-test plot handles empty call" {
      local got="$( wirecell-test plot 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-util ario-cmp handles empty call" {
      local got="$( wirecell-util ario-cmp 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-util convert-dunevd-wires handles empty call" {
      local got="$( wirecell-util convert-dunevd-wires 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-util convert-icarustpc-wires handles empty call" {
      local got="$( wirecell-util convert-icarustpc-wires 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-util convert-multitpc-wires handles empty call" {
      local got="$( wirecell-util convert-multitpc-wires 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-util convert-oneside-wires handles empty call" {
      local got="$( wirecell-util convert-oneside-wires 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-util convert-uboone-wire-regions handles empty call" {
      local got="$( wirecell-util convert-uboone-wire-regions 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-util frame-split handles empty call" {
      local got="$( wirecell-util frame-split 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-util gen-plot-wires handles empty call" {
      local got="$( wirecell-util gen-plot-wires 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-util gravio handles empty call" {
      local got="$( wirecell-util gravio 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-util ls handles empty call" {
      local got="$( wirecell-util ls 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-util make-map handles empty call" {
      local got="$( wirecell-util make-map 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-util make-wires handles empty call" {
      local got="$( wirecell-util make-wires 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-util make-wires-onesided handles empty call" {
      local got="$( wirecell-util make-wires-onesided 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-util npz-to-img handles empty call" {
      local got="$( wirecell-util npz-to-img 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-util npz-to-wct handles empty call" {
      local got="$( wirecell-util npz-to-wct 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-util plot-select-channels handles empty call" {
      local got="$( wirecell-util plot-select-channels 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-util plot-wire-regions handles empty call" {
      local got="$( wirecell-util plot-wire-regions 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-util plot-wires handles empty call" {
      local got="$( wirecell-util plot-wires 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-util wire-channel-map handles empty call" {
      local got="$( wirecell-util wire-channel-map 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-util wire-summary handles empty call" {
      local got="$( wirecell-util wire-summary 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-util wires-channels handles empty call" {
      local got="$( wirecell-util wires-channels 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-util wires-info handles empty call" {
      local got="$( wirecell-util wires-info 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-util wires-ordering handles empty call" {
      local got="$( wirecell-util wires-ordering 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-util wires-volumes handles empty call" {
      local got="$( wirecell-util wires-volumes 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-validate diff-hists handles empty call" {
      local got="$( wirecell-validate diff-hists 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-validate magnify-diff handles empty call" {
      local got="$( wirecell-validate magnify-diff 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-validate magnify-dump handles empty call" {
      local got="$( wirecell-validate magnify-dump 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-validate magnify-jsonify handles empty call" {
      local got="$( wirecell-validate magnify-jsonify 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-validate magnify-plot handles empty call" {
      local got="$( wirecell-validate magnify-plot 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-validate magnify-plot-reduce handles empty call" {
      local got="$( wirecell-validate magnify-plot-reduce 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

@test "assure wirecell-validate npz-load handles empty call" {
      local got="$( wirecell-validate npz-load 2>&1 )"
      [[ -z "$( echo $got | grep Traceback)" ]]
}

