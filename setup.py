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
            'wirecell-sigproc = wirecell.sigproc.__main__:main',
            'wirecell-util = wirecell.util.__main__:main',
            'wirecell-gen = wirecell.gen.__main__:main',
            'wirecell-validate = wirecell.validate.__main__:main',
            'wirecell-pgraph = wirecell.pgraph.__main__:main',
            'wirecell-img = wirecell.img.__main__:main',
            'wirecell-resp = wirecell.resp.__main__:main',
            'wirecell-plot = wirecell.plot.__main__:main',
            'wirecell-pytorch = wirecell.pytorch.__main__:main',
            'wirecell-aux = wirecell.aux.__main__:main',
        ]
    )
)

