#!/usr/bin/env python3
'''
Some helpers for __main__.py CLIs
'''
from wirecell.util import jsio
import click
import functools

# The jsonnet_loader() decorator provides common CLI args and handling
# to provide a Click command function body a pre-loaded object from a
# named function.  Replace @click.argument("myfilenamearg") and
# subsequent explicit handling with:
#
# from wirecell.util.cli import jsonnet_loader
# from wirecell.util import jsio
# 
# @click.command()
# @jsonnet_loader("myfilenamearg")
# def mycmd(myfilenamearg):
#     ### use myfilenamearg directly as object
def jsonnet_loader(jfilekey):
    def decorator(func):
        @click.option("-J", "--jpath", multiple=True,
                      envvar='WIRECELL_PATH', 
                      help="A file system path to locate Jsonnet files")
        @click.option("-A", "--tla", multiple=True,
                      help="Set a top-level argument as key=val, key=code or key=filename")
        @click.option("-V", "--ext", multiple=True,
                      help="Set an external var (avoid this with new jsonnet code)")
        @click.argument(jfilekey)
        @functools.wraps(func)
        def wrapper(*args, **kwds):
            # print(args)
            # print(kwds)
            jfkey = jfilekey.replace("-","_")
            jfile = kwds.pop(jfkey)
            jpath = jsio.wash_path(kwds.pop("jpath"))            
            tla = kwds.pop("tla")
            jkwds = jsio.tla_pack(tla, jpath)
            # print(jkwds)
            ext = kwds.pop("ext")
            jkwds.update(jsio.tla_pack(ext, jpath, 'ext_'))
            kwds[jfkey] = jsio.load(jfile, jpath, **jkwds)
            return func(*args, **kwds)
        return wrapper
    return decorator

@click.command()
@jsonnet_loader("testfile")
def testcli(testfile):
    print(type(testfile), len(testfile))
if '__main__' == __name__:
    testcli()
    
