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
        'pytest',
        'numpy',
        'matplotlib',
        'networkx',
        'gojsonnet',
        'semver',
        'sqlalchemy',
        'semver',
        'scipy',
        'moo @ git+https://github.com/brettviren/moo.git@0.6.6#egg=moo',
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
            'wirecell-test = wirecell.test.__main__:main',
            'wirecell-ls4gan = wirecell.ls4gan.__main__:main',
        ]
    )
)

