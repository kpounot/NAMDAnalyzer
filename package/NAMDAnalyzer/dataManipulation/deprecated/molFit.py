import os, sys
import numpy as np

from scipy.optimize import minimize

def molRotation(theta, phi, nu):
    """ This defines the rotation matrix that will be applied to molecules coordinates. """

    cosTheta    = np.cos(theta)
    sinTheta    = np.sin(theta)
    cosPhi      = np.cos(phi)
    sinPhi      = np.sin(phi)
    cosNu       = np.cos(nu)
    sinNu       = np.sin(nu)

    rotX = np.array([   [1,         0,          0        ], 
                        [0,         cosTheta,   -sinTheta], 
                        [0,         sinTheta,   cosTheta ] ])

    rotY = np.array([   [cosPhi,    0,          sinPhi   ], 
                        [0,         1,          0        ], 
                        [-sinPhi,   0,          cosPhi   ] ])

    rotZ = np.array([   [cosNu,     -sinNu,     0        ], 
                        [sinNu,     cosNu,      0        ], 
                        [0,         0,          1        ] ])

    return rotX, rotY, rotZ


def costFunc(params, firstFrame, frameData):
    """ Computes the cost - deviation from first frame - for the current frame with given rotation
        angles in 'params' argument. """

    rotX, rotY, rotZ = molRotation(*params)

    frameData = np.dot(rotX, frameData.T).T
    frameData = np.dot(rotY, frameData.T).T
    frameData = np.dot(rotZ, frameData.T).T

    return np.sum( (frameData - firstFrame)**2 )


def alignAllMol(dataSet, centerOfMass):
    """ This function takes trajectories from .dcd file, and apply the following procedure 
        for all frame between begin and end and to all atoms between firstAtom and lastAtom:

        - move center of mass of each molecule to the origin
        - rotate every molecule to get the best fit with the first frame
        - return the resulting trajectories on the same format as the initial dataSet """


    #_Substract the center of mass coordinates to each atom for each frame,
    for i in range(dataSet.shape[1]):
        dataSet[:,i,:] = dataSet[:,i,:] - centerOfMass[:,i]

    #_Determines the correct rotations to align the molecules
    for i in range(1, dataSet.shape[1]):

        params = minimize(  costFunc, 
                            [1, 1, 1],
                            args=(dataSet[:,0,:], dataSet[:,i,:]) )

        #_Apply the rotation using the fitted parameters
        rotM = molRotation(*params.x)
        dataSet[:,i,:] = np.dot( rotM[0], dataSet[:,i,:].T ).T
        dataSet[:,i,:] = np.dot( rotM[1], dataSet[:,i,:].T ).T
        dataSet[:,i,:] = np.dot( rotM[2], dataSet[:,i,:].T ).T


    return dataSet
            
