#!/usr/bin/env bats

# This exercises the detlinegen and linegen commands of wirecell-gen.

bats_load_library "wct-bats.sh"

function det_make_plots () {
    local det="$1" ; shift
    local pln="$1" ; shift
    local theta_y="$1" ; shift
    local theta_xz="$1" ; shift
    local coords="${1:-global}"; shift
    
    local apa="0"

    local title="${det} apa${apa} plane${pln} y${theta_y} xz${theta_xz} $coords"
    local name="$(echo "$title" | tr ' ' '-' | tr -d '*')"

    echo "$title"

    wcpy gen detlinegen -d $det --apa $apa --plane $pln \
         --theta_y="$theta_y" --theta_xz="$theta_xz" \
         --angle-coords="$coords" \
         -o "depos-$name.npz" -m "meta-$name.json"

    wcpy gen plot-depos -p qxz -p qxy -p qzy --title "$title" \
         -o "depos-${name}.pdf" "depos-$name.npz"

}

@test "detlinegen plots" {
    cd_tmp file


    for det in pdsp # uboone
    do

        # Some care in iterating over angles is needed.  If the track is
        # parallel to a global coordinate the plotter will barf.
        for theta_y in '10*deg' '80*deg'
        do
            for theta_xz in '10*deg' '80*deg'
            do

                # can skip global as it is same as wire-plane 2.
                # det_make_plots "$det" "2" "$theta_y" "$theta_xz" "global"
                for pln in 0 1 2
                do
                    det_make_plots "$det" "$pln" "$theta_y" "$theta_xz" "wire-plane"
                done
            done
        done
    done
}
