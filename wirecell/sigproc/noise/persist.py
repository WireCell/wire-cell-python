import json

from schema import NoiseSpectrum

# fixme: content here and in other "persist" submodules is repetitive and
# should be factored to common ground.

def dumps(spectra, indent=4):
    '''
    Dump the spectra to a JSON string
    '''
    spectra = [s._asdict() for s in spectra]
    return json.dumps(spectra, indent=indent)

def loads(text):
    '''
    Return a list of NoiseSpectrum objects from the JSON text
    '''
    spectra = json.loads(text)
    return [NoiseSpectrum(**s) for s in spectra]


def dump(filename, spectra):
    '''
    Save a list of wirecell.sigproc.noise.NoiseSpectrum objects to a
    file of the given name.

    File is saved depending on extension.  .json, .json.bz2 and
    .json.gz are supported.
    '''
    text = dumps(spectra,indent=4)
    if filename.endswith(".json"):
        open(filename, 'w').write(text)
        return
    if filename.endswith(".json.bz2"):
        import bz2
        bz2.BZ2File(filename, 'w').write(text)
        return
    if filename.endswith(".json.gz"):
        import gzip
        gzip.open(filename, "wb").write(text)
        return
    raise ValueError("unknown file format: %s" % filename)

def load(filename):
    '''
    Return a list of wirecell.sigproc.noise.NoiseSpectrum objects
    loaded from the given file.

    File is loaded depending on extension.  .json, .json.bz2 and
    .json.gz are supported.
    '''
    if filename.endswith(".json"):
        return loads(open(filename, 'r').read())

    if filename.endswith(".json.bz2"):
        import bz2
        return loads(bz2.BZ2File(filename, 'r').read())

    if filename.endswith(".json.gz"):
        import gzip
        return loads(gzip.open(filename, "rb").read())

    raise ValueError("unknown file format: %s" % filename)
