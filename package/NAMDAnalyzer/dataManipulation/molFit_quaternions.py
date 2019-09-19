import os, sys
import numpy as np

from scipy.optimize import newton

#_Quaternion multiplication
def q_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

    return w, x, y, z



def q_conjugate(q):
    w, x, y, z = q

    return (w, -x, -y, -z)




def qv_mult(v1, q1):
    q2 = np.insert(v1, 0, [0.0])

    return q_mult(q_mult(q1, q2), q_conjugate(q1))[1:]



def get_bestMatrix(q):
    w, x, y, z = q

    qM = np.array(  [ [w, -x, -y, -z],
                      [x,  w,  z, -y],
                      [y, -z,  w,  x],
                      [z,  y, -x,  w] ] )

    return qM


def alignAllMol(dcdData):
    """This function takes trajectories from :class:`.Dataset` *dcdData* attribute, 
    and apply the following procedure for each frames in *dcdData* array:

    First, computes appropriate dot products are computed with the first frames coordinates
    Then, the matrix to be solved for eigenvalues and eigenvectors is constructed and solved


    :arg dcdData: Array containing trajectory coordinates for wanted frames
                  with shape (atoms, frames, coordinates)
    :type dcdData: :class:`.numpy.ndarray`


    :returns: A list of 4 by 4 rotation matrices to be applied on *dcdData* array

    
    This procedure is based on the [Horn_1987]_ and [Theobald_2005]_ papers.

    References:

    .. [Horn_1987] https://doi.org/10.1364/JOSAA.4.000629
    .. [Theobald_2005] https://doi.org/10.1107/S0108767305015266

    """


    qM = []

    #_Determines the correct rotations to align the molecules
    for i in range(1, dcdData.shape[1]):
        #_Here comes a long series of coefficient calculation
        S_xx = np.dot(dcdData[:,i,0].T, dcdData[:,0,0]).T
        S_xy = np.dot(dcdData[:,i,0].T, dcdData[:,0,1]).T
        S_xz = np.dot(dcdData[:,i,0].T, dcdData[:,0,2]).T
        S_yx = np.dot(dcdData[:,i,1].T, dcdData[:,0,0]).T
        S_yy = np.dot(dcdData[:,i,1].T, dcdData[:,0,1]).T
        S_yz = np.dot(dcdData[:,i,1].T, dcdData[:,0,2]).T
        S_zx = np.dot(dcdData[:,i,2].T, dcdData[:,0,0]).T
        S_zy = np.dot(dcdData[:,i,2].T, dcdData[:,0,1]).T
        S_zz = np.dot(dcdData[:,i,2].T, dcdData[:,0,2]).T

        yyzz__yzzy  = (S_yy*S_zz - S_yz*S_zy)
        xx__yy__zz  = (S_xx - S_yy - S_zz)
        xx__yy_zz   = (S_xx - S_yy + S_zz)
        xx_yy__zz   = (S_xx + S_yy - S_zz)
        xx_yy_zz    = (S_xx + S_yy + S_zz)
        xz_zx       = (S_xz + S_zx)
        xy_yx       = (S_xy + S_yx)
        xy__yx      = (S_xy - S_yx)
        xz__zx      = (S_xz - S_zx)
        yx_xy       = (S_yx + S_xy)
        yz_zy       = (S_yz + S_zy)
        yx__xy      = (S_yx - S_xy)
        yz__zy      = (S_yz - S_zy)
        yx_xy       = (S_yx + S_xy)
        yz_zy       = (S_yz + S_zy)
        yx__xy      = (S_yx - S_xy)
        yz__zy      = (S_yz - S_zy)
        zx_xz       = (S_zx + S_xz)
        zy_yz       = (S_zy + S_yz)
        zx__xz      = (S_zx - S_xz)
        zy__yz      = (S_zy - S_yz) 
        
        M = np.array([  [xx_yy_zz,      yz__zy,         zx__xz,     xy__yx],
                        [yz__zy,        xx__yy__zz,     xy_yx,      zx_xz],
                        [zx__xz,        xy_yx,          -xx__yy_zz, yz_zy],
                        [xy__yx,        zx_xz,          yz_zy,      -xx_yy__zz] ])

        eigval, eigvec = np.linalg.eig(M)

        bestVec = eigvec[:,np.argwhere(eigval == np.max(eigval))[0][0]]

        #_Obtain the rotation matrix
        qM.append(get_bestMatrix(bestVec))


    return qM
            


def applyRotation(dcdData, qM):
    """ Apply a rotation using given rotation matrix qM on given dcd data. """

    for i in range(1, dcdData.shape[1]):

        #_Generating a data matrix with extra column containing zeros
        tempData = np.zeros( (dcdData[:,i:].shape[0], dcdData[:,i,:].shape[1] + 1) )
        tempData[:, 1:] = dcdData[:,i,:]

        #_Applying the rotation
        tempData = np.dot(qM[i], tempData.T).T

        #_Writing the data back in dcdData
        dcdData[:,i,:] = tempData[:, 1:]


    return dcdData


