import numpy

def median(frame):
    '''
    Return median-subtracted frame
    '''
    fmed = numpy.median(frame, axis=1)
    return (frame.T - fmed).T
    
def mean(frame):
    '''
    Return mean-subtraced frame.
    '''
    fmu = numpy.mean(frame, axis=1)
    return (frame.T - fmu).T

def ac(frame):
    '''
    Return AC-coupled frame
    '''
    cspec = numpy.fft.fft(frame)
    cspec[:,0] = 0      # set all zero freq bins to zero
    wave = numpy.fft.ifft(cspec)
    return numpy.real(wave)

def none(frame):
    return frame

