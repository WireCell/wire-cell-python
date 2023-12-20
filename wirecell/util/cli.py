#!/usr/bin/env python3
'''
CLI decorators to help build __main__'s

'''
from wirecell.util import jsio, ario, plottools

import click
import functools

## All wire-cell-python "should" use logging, not print()
import logging
log = logging.getLogger('wirecell')
debug = log.debug
info = log.info
warn = log.warn
warning = log.warning
critical = log.critical

# Every wire-cell-python __main__.py must use this to define the command group.
def context(group_name, log_name="wirecell"):
    '''
    Add "standard" base options and set up logging.

    Usage from a __main__.py:

    To make the Click group, pass the "short name" of the WCT pkg as the group_name.  Eg "img" or "util":

        from wirecell.util.cli import context, log
        @context("grp")
        def cli(ctx):
            """
            Wire-Cell command for grp
            """
            pass

    To use logging in a command:

        @cli.command("my-command")
        def my_command():
            """
            My docstring
            """
            log.debug("some debug message")

    To use logging in a module

        import logging
        log = logging.getLogger(__name__)
        def foo():
            log.debug("some message")

    Replace all print() with log.

    '''

    def decorator(func):
        cmddef = dict(context_settings = dict(help_option_names=['-h', '--help']))
        @click.group(group_name, **cmddef)
        @click.option("-l","--log-output", multiple=True,  help="log to a file [default:stdout]")
        @click.option("-L","--log-level", default="info", help="set logging level [default:info]")
        @click.pass_context
        @functools.wraps(func)
        def wrapper(ctx, log_output, log_level, *args, **kwds):
            '''
            Wire-Cell Toolkit command 
            '''
            log = logging.getLogger(log_name)
            try:
                level = int(log_level)      # try for number
            except ValueError:
                level = log_level.upper()   # else assume label
            log.setLevel(level)

            if not log_output:
                log_output = ["stdout"]
            for one in log_output:
                if one == "stdout":
                    sh = logging.StreamHandler()
                    sh.setLevel(level)
                    log.addHandler(sh)
                    continue
                fh = logging.FileHandler(one)
                fh.setLevel(level)
                log.addHandler(fh)
            return
        return wrapper
    return decorator
        
        
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
#
# Or, to make use of registry indirection
# @click.command()
# @jsonnet_loader("myfilenamearg", "wires")
# def mycmd(myfilenamearg):
#     ### use myfilenamearg directly as object
#
# Note for CLI user, -A/-V values are interpreted as string, code or a filename.
#
# -A foo=string
# -A bar='["list","of","string","code"]'
# -A file=more.jsonnet
# -A label='"string as code"'
# -A slurp="$(cat somedata.txt | convert-to-json)"
#
# When hacking at this TLA/EXT interface, be wary of shell quoting rules!
def jsonnet_loader(jfilekey, regkey=None):
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
            if '.json' in jfile or regkey is None:
                kwds[jfkey] = jsio.load(jfile, paths=jpath, **jkwds)
            else:
                kwds[jfkey] = jsio.load_registry(jfile, regkey=regkey, paths=jpath, **jkwds)
            return func(*args, **kwds)
        return wrapper
    return decorator


    
def frame_input(suffix=""):
    '''Return a decorator for a command taking a frame as input.

    For multi-frame input, pass a unique suffix which will be appended
    to command line flag and keyword identifiers.

    '''

    def frame_input_one(func):
        '''
        A decorator for a command that inputs a frame file.

        Provides arguments to command:

        - array :: the frame array object
        - aname :: the name of the frame array
        - ariofile :: the ario file object

        '''
        import numpy

        dfix = suffix.replace("_","-")
        ufix = suffix.replace("-","_")

        @click.option("--ident"+dfix, default=None, type=str, help="locate frame with ident [default=first]")
        @click.option("--tier"+dfix, default='*', type=str, help="locate frame array by data tier [default='*']")
        @click.option("--frame"+dfix, default=None, type=str, help="explicitly name frame array")
        @click.argument("ariofile"+dfix)
        @functools.wraps(func)
        def wrapper(*args, **kwds):

            fname = kwds.pop("ariofile"+ufix)
            fp = ario.load(fname)
            kwds["ariofile"+ufix] = fp

            def no_frame(msg):
                have = '", "'.join(fp.keys())
                raise click.BadParameter(f'{msg}: have keys: "{have}"')

            frame_name = kwds.pop("frame"+ufix, None)
            if frame_name is None:
                fnames = [key for key in fp.keys() if key.startswith("frame_")]
                if not fnames:
                    no_frame("no frame key")

                tier = kwds.pop("tier"+ufix, '*')
                if not tier: tier = '*'
                if tier != '*':
                    fnames = [f for f in fnames if f.startswith(f'frame_{tier}')]
                ident = kwds.pop("ident"+ufix, None)
                if ident is None:
                    frame_name = fnames[0]
                else:
                    fnames = [f for f in fnames if f.endswith(f'_{ident}')]
                    if not fnames:
                        no_frame(f'No matching frame with tier={tier}, ident={ident}')
                    frame_name = fnames[0]

            if frame_name not in fp:
                no_frame(f'array "{aname}" not in "{fname}"')

            _, tier, ident = frame_name.split("_")

            arr = fp[frame_name]
            kwds["array"+ufix] = arr
            kwds["aname"+ufix] = frame_name;
            kwds["tier"+ufix] = tier
            kwds["ident"+ufix] = ident
            kwds["channels"+ufix] = fp[f'channels_{tier}_{ident}']
            kwds["tickinfo"+ufix] = fp[f'tickinfo_{tier}_{ident}']

            return func(*args, **kwds)
        return wrapper
    return frame_input_one


def image_output(func):
    '''
    A decorator for a command that outputs some kind of matplotlib image.

    Provides arguments to command:

    - cmap :: a color map object
    - vmin :: a minimum value
    - vmax :: a maximum value
    - output :: an object with .savefig() method.

    '''

    from matplotlib import colormaps

    @click.option("--cmap", default="viridis", help="Color map name [default=viridis]")
    @click.option("--vmin", type=float, default=None, help="Set min value [default=None]")
    @click.option("--vmax", type=float, default=None, help="Set max value [default=None]")
    @click.option("--format", default=None, help="Output file format, [default=auto]")
    @click.option("-o", "--output", default=None, help="Output file, [default=stdout]")
    @click.option("--single", is_flag=True, default=False,
                  help="Force a single plot [default=False]")
    @click.option("--dpi", default=150,
                  help="Image resolution in dots-per-inch [default=150]")
    @functools.wraps(func)
    def wrapper(*args, **kwds):
        fmt = kwds["format"]
        output = kwds["output"]
        if fmt is None:
            if output is None or len(output.split(".")) == 1:
                fmt = "png"
            else:
                fmt = output.split(".")[-1]

        if output is None:
            output = "/dev/stdout"

        single = kwds.pop("single", None)
        if single:
            kwds["output"] = plottools.NameSingleton(output, format=fmt)
        else:
            kwds["output"] = plottools.pages(output, format=fmt)

        kwds["cmap"] = colormaps[kwds["cmap"]]

        return func(*args, **kwds)
    return wrapper




@click.command()
@jsonnet_loader("testfile")
def testcli(testfile):
    print(type(testfile), len(testfile))
if '__main__' == __name__:
    testcli()
    
