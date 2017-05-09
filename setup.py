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
    ],
    entry_points = dict(
        console_scripts = [
            'wirecell-sigproc = wirecell.sigproc.main:main',
            'wirecell-util = wirecell.util.main:main',
        ]
    )
)

