import numpy

def pca_eigen(pts):
    '''
    Return the (eigenvectors, eigenvalues) of the batch of points in descending order of eignevalue.
    '''
    pts_std = (pts - pts.mean(axis=0)) / pts.std(axis=0)
    cov_matrix = numpy.cov(pts_std, rowvar=False)
    eigenvalues, eigenvectors = numpy.linalg.eig(cov_matrix)
    sorted_index = numpy.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:,sorted_index]
    sorted_eigenvalues = eigenvalues[sorted_index]
    return sorted_eigenvectors, sorted_eigenvalues

