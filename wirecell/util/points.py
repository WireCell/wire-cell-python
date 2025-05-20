import numpy

def pca_eigen(pts):
    '''
    Return the (eigenvectors, eigenvalues) of the batch of points in descending order of eignevalue.
    '''
    variances = numpy.var(pts, axis=0)

    varied_axes = variances > 0

    # Remove features with zero variance
    pts_filtered = pts[:, varied_axes]


    if pts_filtered.shape[1] == 0:
        return ([numpy.zeros_like(pts[0])], [0])

    # Center the pts
    mean = numpy.mean(pts_filtered, axis=0)
    centered_pts = pts_filtered - mean

    # Calculate the covariance matrix
    covariance_matrix = numpy.cov(centered_pts, rowvar=False)


    # Calculate eigenvalues and eigenvectors
    try:
        eigenvalues, eigenvectors = numpy.linalg.eig(covariance_matrix)
    except numpy.linalg.LinAlgError:
        return ([numpy.zeros_like(pts[0])], [0])
        

    # Sort eigenvalues and eigenvectors in descending order
    idx = numpy.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]


    return eigenvectors, eigenvalues



    # std = pts.std(axis=0)
    # if 0 in std:
        

    # pts_std = (pts - pts.mean(axis=0)) / std
    # cov_matrix = numpy.cov(pts_std, rowvar=False)
    # eigenvalues, eigenvectors = numpy.linalg.eig(cov_matrix)
    # sorted_index = numpy.argsort(eigenvalues)[::-1]
    # sorted_eigenvectors = eigenvectors[:,sorted_index]
    # sorted_eigenvalues = eigenvalues[sorted_index]
    # return sorted_eigenvectors, sorted_eigenvalues

