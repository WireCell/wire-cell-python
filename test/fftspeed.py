import numpy

def roundtrip(wave, size):
    nchan, ntick = wave.shape

    if size == ntick:
        return wave

    spec = numpy.fft.fft(wave, axis=1)
    spec = numpy.fft.ifftshift(spec)

    # note: this is not quite right but good enough for speed
    if size < ntick:            # downsample
        half = (ntick-size)//2
        spec = spec[:, half:-half]
    else:                       # upsample
        half = (size-ntick)//2
        spec = numpy.hstack( (numpy.zeros((nchan,half), dtype=spec.dtype),
                              spec,
                              numpy.zeros((nchan,half), dtype=spec.dtype)) )
    spec = numpy.fft.fftshift(spec)
    wave = numpy.real(numpy.fft.ifft(spec))
    return wave * wave.shape[1] / ntick

import timeit
def main(filename, size):
    f = numpy.load(filename)
    fr = f[f.files[0]]
    
    def runit():
        roundtrip(fr, size)

    got = timeit.timeit(runit, number=10)
    print(got)

if '__main__' == __name__:
    import sys
    try:
        size = int(sys.argv[2])
    except IndexError:
        size = 6144

    main(sys.argv[1], int(sys.argv[2]))

