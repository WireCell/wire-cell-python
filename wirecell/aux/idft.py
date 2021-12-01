import os
import json
import numpy
import matplotlib.pyplot as plt
from collections import defaultdict
from wirecell.util import ario
import tarfile

def load(filename):
    dat = json.loads(open(filename).read())
    return dat;

def gen_arrays():
    '''
    Return dictionary of a few "canned" arrays
    '''
    ret = {
        "rand1d": numpy.array(numpy.random.random((64,)), dtype='f4'),
        "rand2d": numpy.array(numpy.random.random((8,8)), dtype='f4'),
        "imp1d": numpy.zeros(8*3, dtype='f4'),
        "imp2d": numpy.zeros((8,3), dtype='f4'),
    }
    ret["imp1d"][0] = 1.0
    ret["imp2d"][1,0] = 1.0
    return ret

def gen_config(filename):
    ret = [
        {
		"dst" : "rand1d-fwd1d_r2c",
		"op" : "fwd1d_r2c",
		"src" : "rand1d"
	},
	{
		"dst" : "rand2d-fwd2d_r2c",
		"op" : "fwd2d_r2c",
		"src" : "rand2d"
	},
	{
		"dst" : "imp1d-fwd1d_r2c",
		"op" : "fwd1d_r2c",
		"src" : "imp1d"
	},
	{
		"dst" : "imp2d-fwd2d_r2c",
		"op" : "fwd2d_r2c",
		"src" : "imp2d"
	}
    ];
    open(filename, "w").write(json.dumps(ret, indent=4));
    return ret;


def get_arrays(filelst = None):
    '''
    Return a dictionary of arrays from file names in filelst which is
    expected to be a list of tar[.bz2|.gz] files.
    '''
    if not filelst:
        return gen_arrays()
    ret = dict()
    for fname in filelst:
        print(f'loading {fname}')
        reader = ario.load(fname)
        for key in reader.keys():
            val = reader[key]
            print(f'{key}: {type(val)}')
            ret[key] = reader[key]
    return ret

def save_arrays(fname, arrays):
    '''
    Save arrays to tar stream fname.

    Array names will have .npy appended if missing.
    '''
    # fixme: it would be nice to add this to ario
    mode = "w"
    if fname.endswith('.bz2'): mode += ":bz2"
    if fname.endswith('.gz'): mode += ":gz"
    tf = tarfile.open(fname, mode=mode, format=tarfile.GNU_FORMAT)
    for name, arr in arrays.items():
        # fixme: use tempfile
        numpy.save("tmp.npy", arr);
        if not name.endswith(".npy"):
            name += ".npy"
        tf.add("tmp.npy", name);
        os.unlink("tmp.npy")
    tf.close()

def select_array(dat, func_name, first=False, inplace=True):
    '''
    Return dictionary of matched arrays.  Keys:

    size, nrows, ncols, time, clock

    Arrays are ordered by size = nrows*ncols
    '''
    ret = defaultdict(list)

    for one in dat:
        if one.get("func", "") != func_name:
            continue
        if one.get("first", None) != first:
            continue
        if one.get("in-place", None) != inplace:
            continue
        sw = one["stopwatch"]

        nrows = one["nrows"]
        ncols = one["ncols"]
        size = nrows*ncols
        ret["nrows"].append(nrows)
        ret["ncols"].append(ncols)
        ret["size"].append(size)
        ret["ntimes"].append(one["ntimes"])
        ret["clock"].append(sw["clock"]["elapsed"])
        ret["time"].append(sw["time"]["elapsed"])

    sinds = numpy.argsort(ret["size"])
    return {k:numpy.array(v)[sinds] for k,v in ret.items()}

def label(dat):
    d = dat[0]
    si = d['sysinfo']
    lab = d['typename']
    # fixme: for now, assume any config means GPU, which is NOT right in general
    if d['config']:
        gpu = si['gpus'][0]
        lab += ' ' + gpu
    else:
        cpu = si['cpu']['model'].replace('Intel(R) Core(TM) ', '')
        lab += ' ' + cpu
    return lab

def plot_init(tit, xtit, ytit):
    fig = plt.figure(figsize=(10,9))
    ax = plt.subplot()
    ax.grid(linestyle="--", linewidth=0.5, color=".25", zorder=-10)
    ax.set_title(tit, fontsize=20, verticalalignment="bottom")
    ax.set_xlabel(xtit)
    ax.set_ylabel(ytit)
    return fig,ax

def plot_time(dats, func_name, measure='time'):
    fig, ax = plot_init(f'"{func_name}" transform, measure: "{measure}"',
                        "size of array [N]", f"{measure}/size [ns/N]")

    # fig = plt.figure(figsize=(10,9))
    # ax = plt.subplot()
    # ax.grid(linestyle="--", linewidth=0.5, color=".25", zorder=-10)
    # ax.set_title(f'"{func_name}" transform, measure:"{measure}"', fontsize=20, verticalalignment="bottom")
    # ax.set_xlabel("size of array [N]")
    # ax.set_ylabel("time/size [ns/N]")
    for ind, dat in enumerate(dats):
        arr = select_array(dat, func_name)
        # print (ind, func_name, list(arr.keys()))
        x = arr['size']
        y = arr[measure]/(arr['size']*arr['ntimes'])
        ax.plot(x, y, marker='o', label=label(dat))
    ax.legend()

def plot_plan_time(dats, func_name, measure='time'):
    fig, ax = plot_init(f'"{func_name}" cold/warm, measure: "{measure}"',
                        "size of array [N]", f"{measure} cold/warm ratio")
    ax.set_ylim(0,100)
    if "2d" in func_name or "1b" in func_name:
        ax.set_ylim(0,2)        
    for ind, dat in enumerate(dats):
        cold = select_array(dat, func_name, first=True)
        warm = select_array(dat, func_name)
        x = cold['size']
        y = cold[measure]/(warm[measure]/warm['ntimes'])
        ax.plot(x, y, marker='o', label=label(dat))
    ax.legend()
    return



#
# DFT operators as labels
#

def fwd1d(arr):
    return numpy.fft.fft(arr)
def inv1d(arr):
    return numpy.fft.ifft(arr)
def fwd2d(arr):
    return numpy.fft.fft2(arr)
def inv2d(arr):
    return numpy.fft.ifft2(arr)

def fwd1d_r2c(arr):
    return numpy.fft.fft(numpy.array(arr, dtype='c8'))
def inv1d_c2r(arr):
    return numpy.real(numpy.fft.ifft(arr))
def fwd2d_r2c(arr):
    return numpy.fft.fft2(numpy.array(arr, dtype='c8'))
def inv2d_c2r(arr):
    return numpy.real(numpy.fft.ifft2(arr))

def fwd1b0(arr):
    return numpy.fft.fft2(arr, axis=0)
def fwd1b1(arr):
    return numpy.fft.fft2(arr, axis=1)
def inv1b0(arr):
    return numpy.fft.ifft2(arr, axis=0)
def inv1b1(arr):
    return numpy.fft.ifft2(arr, axis=1)
