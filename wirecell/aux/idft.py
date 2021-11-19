import json
import numpy
import matplotlib.pyplot as plt
from collections import defaultdict

def load(filename):
    dat = json.loads(open(filename).read())
    return dat;

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

    

    pass
    
