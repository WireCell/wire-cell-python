from wirecell import units
import matplotlib.pyplot as plt
import numpy

def subplots(nrows=1, ncols=1):
    return plt.subplots(nrows, ncols, tight_layout=True)

def blob_nodes(cgraph):
    ret = list()
    for node, ndata in cgraph.nodes.data():
        if ndata['code'] == 'b':
            ret.append(ndata)
    return ret


def blob_coord(blobs, axis=0):
    ret = list()
    for b in blobs:
        ret.append(b['corners'][0][axis])
    return numpy.array(ret)
     

def blob_centers(blobs, index=2):
    ret = list()
    for b in blobs:
        c = 0
        corners = b['corners']
        for corner in corners:
            c += corner[index]
        cen = c/len(corners)
        ret.append(cen)
    return numpy.array(ret)

def plot_xz(depos, cgraph):
    fig, ax = subplots(1,1)
    blobs = blob_nodes(cgraph)
    ax.scatter(blob_coord(blobs)/units.cm, blob_centers(blobs)/units.cm,
               marker='s', label="blobs")
    ax.scatter(depos['x']/units.cm, depos['z']/units.cm,
               marker='.', label="depos")

    ax.set_title("depos and blobs")
    ax.set_xlabel("X [cm]")
    ax.set_ylabel("Z [cm]")    
    ax.legend()
    return fig
