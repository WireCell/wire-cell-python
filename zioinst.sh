#!/bin/bash
#
# Install WCT related Python.
# Run from source directory and enter a venv first.

mydir="$(dirname $(realpath $BASH_SOURCE))"
if [ ! -f "$mydir/wirecell/units.py" ] ; then
    cat <<EOF
Refusing to run self not in source directory:

   wire-cell-toolkit/python/ 

Do not move me.
EOF
    exit -1
fi


if [ -z "$VIRTUAL_ENV" ] ; then
    cat <<EOF
Refusing to install outside of a virtual environment.
To make one do this:

  python3 -m venv /path/to/venv
  source /path/to/venv/bin/activate

Then, rerun this script.
EOF
    exit -1
fi


if [ ! -x "$(dirname $(which python))/ipython" ] ; then
    pip install ipython
else
    echo "have ipython"
fi

if ! python -c 'import zmq' > /dev/null 2>&1 ; then
    prefix="$(pkg-config libzmq --variable=prefix)"
    if [ -d "$prefix" ] ; then
        pip install --pre pyzmq --install-option=--enable-drafts --install-option=--zmq=$prefix
    else
        echo "Need libzmq"
        exit -1
    fi
else
    echo "have PyZmq"
fi

if ! python -c 'import pyre' > /dev/null 2>&1 ; then
    pip install 'https://github.com/zeromq/pyre/archive/51451524f0107b67a8e1235c9d85e364d898657a.zip#egg=zio-0.0.0'
else
    echo "have Pyre"
fi

if ! python -c 'import rule' > /dev/null 2>&1 ; then
    pip install 'git+https://github.com/brettviren/rule.git#egg=rule-0.1.2bv'
else
    echo "have rule (bv hacked version)"
fi

if ! python -c 'import zio' > /dev/null 2>&1 ; then
    pip install -e 'git+https://github.com/brettviren/zio.git#egg=zio-0.0.0&subdirectory=python'
else
    echo "have zio"
fi

if ! python -c 'import wirecell' > /dev/null 2>&1 ; then
    pushd $mydir
    pip install -e .
    popd
fi
