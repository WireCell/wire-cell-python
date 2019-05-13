#!/usr/bin/env python
'''
Build/install Wire Cell Toolkit Python

FIXME: this currently is broken if installed in the wider context of a
top-level wirecell package.
'''

from setuptools import setup, find_packages
setup(
    name = 'wirecell',
    version = '0.0',
    packages = find_packages(),
    install_requires = [
        'Click',
        'numpy',
        'matplotlib',
        'mayavi',
        'networkx',
    ],
    entry_points = dict(
        console_scripts = [
            'wirecell-sigproc = wirecell.sigproc.main:main',
            'wirecell-util = wirecell.util.main:main',
            'wirecell-gen = wirecell.gen.main:main',
            'wirecell-validate = wirecell.validate.main:main',
            'wirecell-pgraph = wirecell.pgraph.main:main',
            'wirecell-img = wirecell.img.main:main',
        ]
    )
)

