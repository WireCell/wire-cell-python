#!/usr/bin/env bash

# In order to put each test in its own BATS @test function we discover
# the functions and subfunction.

testdir="$(dirname $(realpath $BASH_SOURCE))"
topdir="$(dirname $testdir)"
testfile="$testdir/test-empty-commands.bats"

cat <<EOF > $testfile
#!/usr/bin/env bats

# This file is generated.  Edits may be lost.
# See: $BASH_SOURCE

EOF

for mainpy in $topdir/wirecell/*/__main__.py
do
    pkg="$(basename $(dirname $mainpy))"
    cmd="wirecell-$pkg"
    for scmd in $( $cmd | awk 'f;/Commands:/{f=1}' | awk '{print $1}' )
    do
        echo "$cmd $scmd"
        cat <<EOF >> $testfile
@test "assure $cmd $scmd handles empty call" {
      local got="\$( $cmd $scmd 2>&1 )"
      [[ -z "\$( echo \$got | grep Traceback)" ]]
}

EOF

        # echo "$cmd $scmd"
        # local got="$( $cmd $scmd 2>&1 )"
        # [[ -z "$( echo $got | grep Traceback )" ]]
    done
done

