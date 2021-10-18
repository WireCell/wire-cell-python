#!/usr/bin/env python
'''
Build/install Wire Cell Toolkit Python

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
        'networkx',
#        'mayavi',
#        'vtk',
    ],
    extras_require = {
        # parse TbbFlow logs and make anigif showing graph states
        'anidfg':  ["GraphvizAnim"] 
    },
    entry_points = dict(
        console_scripts = [
            'wirecell-sigproc = wirecell.sigproc.main:main',
            'wirecell-util = wirecell.util.main:main',
            'wirecell-gen = wirecell.gen.main:main',
            'wirecell-validate = wirecell.validate.main:main',
            'wirecell-pgraph = wirecell.pgraph.main:main',
            'wirecell-img = wirecell.img.main:main',
            'wirecell-resp = wirecell.resp.__main__:main',
        ]
    )
)

