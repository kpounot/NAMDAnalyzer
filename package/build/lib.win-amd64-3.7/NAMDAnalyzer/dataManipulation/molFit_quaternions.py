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
    """ This function takes trajectories from .dcd file, and apply the following procedure 
        for all frame between begin and end and to all atoms between firstAtom and lastAtom:

        - move center of mass of each molecule to the origin
        - rotate every molecule to get the best fit with the first frame
        - return the resulting trajectories on the same format as the initial dcdData 
        
        This procedure is based on the Berthold K. P. Horn (1987) and Douglas L. Theobald (2005) papers."""


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
        qM = get_bestMatrix(bestVec)


        return qM
            


def applyRotation(dcdData, q):
    """ Apply a rotation using given quaternion q on given dcd data. """

    for i in range(1, dcdData.shape[1]):

        #_Obtain the rotation matrix
        q = get_bestMatrix(q)

        #_Generating a data matrix with extra column containing zeros
        tempData = np.zeros( (dcdData[:,i:].shape[0], dcdData[:,i,:].shape[1] + 1) )
        tempData[:, 1:] = dcdData[:,i,:]

        #_Applying the rotation
        tempData = np.dot(qM, tempData.T).T

        #_Writing the data back in dcdData
        dcdData[:,i,:] = tempData[:,1:]


    return dcdData


