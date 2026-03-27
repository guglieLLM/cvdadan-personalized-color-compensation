"""
gradient: Algorithms for Gradient Domain Image Computing

Copyright (C) 2021 Ivar Farup

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or (at
your option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate2d


def diff_filters(diff):
    """
    Compute different forward and backward FDM correlation filters.

    Parameters
    ----------
    diff : str
        finite difference method (FB, cent, Sobel, SobelFB, Feldman, FeldmanFB)

    Returns
    -------
    F_x : ndarray
    F_y : ndarray
    B_x : ndarray
    B_y : ndarray
    """

    if diff == 'FB':
        F_x = np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]])
        F_y = np.array([[0, 1, 0], [0, -1, 0], [0, 0, 0]])
        B_x = np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]])
        B_y = np.array([[0, 0, 0], [0, 1, 0], [0, -1, 0]])
    elif diff == 'cent':
        F_x = .5 * np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])
        F_y = .5 * np.array([[0, 1, 0], [0, 0, 0], [0, -1, 0]])
        B_x = F_x
        B_y = F_y
    elif diff == 'Sobel':
        F_x = .125 * np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        F_y = .125 * np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        B_x = F_x
        B_y = F_y
    elif diff == 'SobelFB':
        F_x = .25 * np.array([[0, -1, 1], [0, -2, 2], [0, -1, 1]])
        F_y = .25 * np.array([[1, 2, 1], [-1, -2, -1], [0, 0, 0]])
        B_x = .25 * np.array([[-1, 1, 0], [-2, 2, 0], [-1, 1, 0]])
        B_y = .25 * np.array([[0, 0, 0], [1, 2, 1], [-1, -2, -1]])
    elif diff == 'Feldman':
        F_x = 1 / 32 * np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])
        F_y = 1 / 32 * np.array([[3, 10, 3], [0, 0, 0], [-3, -10, -3]])
        B_x = F_x
        B_y = F_y
    elif diff == 'FeldmanFB':
        F_x = 1 / 16 * np.array([[0, -3, 3], [0, -10, 10], [0, -3, 3]])
        F_y = 1 / 16 * np.array([[3, 10, 3], [-3, -10, -3], [0, 0, 0]])
        B_x = 1 / 16 * np.array([[-3, 3, 0], [-10, 10, 0], [-3, 3, 0]])
        B_y = 1 / 16 * np.array([[0, 0, 0], [3, 10, 3], [-3, -10, -3]])
    elif diff == 'circFB':
        x = (np.sqrt(2) - 1) / 2
        F_x = (np.array([[0, -x, x], [0, -1, 1], [0, -x, x]]) /
               (2 * x + 1))
        F_y = (np.array([[x, 1, x], [-x, -1, -x], [0, 0, 0]]) /
               (2 * x + 1))
        B_x = (np.array([[-x, x, 0], [-1, 1, 0], [-x, x, 0]]) /
               (2 * x + 1))
        B_y = (np.array([[0, 0, 0], [x, 1, x], [-x, -1, -x]]) /
               (2 * x + 1))
    return F_x, F_y, B_x, B_y


def scale_gradient_linear(u, factor, diff='FB'):
    """
    Compute the scaled gradient of the image with the given factor
    """

    fx, fy, bx, by = diff_filters(diff)

    ux = np.zeros(u.shape)
    uy = np.zeros(u.shape)

    for c in range(3):
        ux[..., c] = correlate2d(u[..., c], fx, 'same', 'symm')
        uy[..., c] = correlate2d(u[..., c], fy, 'same', 'symm')
    
    return factor * ux, factor * uy


def scale_gradient_gamma(u, gamma, diff='FB'):
    """
    Compute the scaled gradient of the image with the given factor
    """

    fx, fy, bx, by = diff_filters(diff)

    ux = np.zeros(u.shape)
    uy = np.zeros(u.shape)

    for c in range(3):
        ux[..., c] = correlate2d(u[..., c], fx, 'same', 'symm')
        uy[..., c] = correlate2d(u[..., c], fy, 'same', 'symm')
    
    return np.abs(ux)**gamma * np.sign(ux), np.abs(uy)**gamma * np.sign(uy)


def diffusion_tensor(u, vx, vy, kappa, diff='FB', isotropic=False, diff_struct=True):
    """
    Compute the (difference) diffusion tensor for the given image.

    Parameters
    ----------
    u : ndarray
        The image
    vx : ndarray
        The x component of the gradient image to be matched
    vy : ndarray
        The y component of the gradient image to be matched
    kappa : float
        The diffusion parameter
    diff : str
        The method to use for computing the finite differences
    isotropic : bool
        Use isotropic instead of anisotropic diffusion
    diff_struct : bool
        Compute the difference diffusion tensor instead of the standard diffusion tensor

    Returns
    -------
    D11, D22, D12 : ndarray
        The components of the diffusion tensor
    """

    if diff_struct:
        vvx = vx
        vvy = vy
    else:
        vvx = np.zeros(vx.shape)
        vvy = np.zeros(vy.shape)

    fx, fy, bx, by = diff_filters(diff)

    gx = np.zeros(u.shape)
    gy = np.zeros(u.shape)

    for c in range(3):
        gx[..., c] = correlate2d(u[..., c], fx, 'same', 'symm')
        gy[..., c] = correlate2d(u[..., c], fy, 'same', 'symm')

    if isotropic:

        gradsq = ((gx - vvx)**2 + (gy - vvy)**2).sum(2)

        D11 = 1 / (1 + gradsq**2 / kappa)
        D22 = D11.copy()
        D12 = np.zeros(D11.shape)
                    
    else:

        S11 = ((gx - vvx)**2).sum(2)
        S12 = ((gx - vvx) * (gy - vvy)).sum(2)
        S22 = ((gy - vvy)**2).sum(2)

        # Eigenvalues and eigenvectors of the structure tensor

        lambda1 = .5 * (S11 + S22 + np.sqrt((S11 - S22)**2 + 4 * S12**2))
        lambda2 = .5 * (S11 + S22 - np.sqrt((S11 - S22)**2 + 4 * S12**2))

        theta1 = .5 * np.arctan2(2 * S12, S11 - S22)
        theta2 = theta1 + np.pi / 2

        v1x = np.cos(theta1)
        v1y = np.sin(theta1)
        v2x = np.cos(theta2)
        v2y = np.sin(theta2)

        # Diffusion tensor

        Dlambda1 = 1 / (1 + lambda1**2 / kappa)
        Dlambda2 = 1 / (1 + lambda2**2 / kappa)

        D11 = Dlambda1 * v1x**2 + Dlambda2 * v2x**2
        D22 = Dlambda1 * v1y**2 + Dlambda2 * v2y**2
        D12 = Dlambda1 * v1x * v1y + Dlambda2 * v2x * v2y

    return D11, D22, D12


def gdip_poisson(u0, vx, vy, nit=501, diff='FB', save=None, save_every=100):
    """
    Gradient domain image processing with the Poisson equation

    Parameters
    ----------
    u0 : ndarray
        The original image
    vx : ndarray
        The x component of the gradient image to be matched
    vy : ndarray
        The y component of the gradient image to be matched
    nit : int
        Number of iterations
    diff : str
        The type of difference convolution filters (see diff_filters)
    save : str
        Filenamebase (e.g., 'im-%03d.png') or None
    save_evry : int
        Save every n iterations

    Returns
    -------
    im : ndarray
        The gradient matched image
    """
    
    fx, fy, bx, by = diff_filters(diff)
    gx = np.zeros(u0.shape)
    gy = np.zeros(u0.shape)

    u = u0.copy()

    if save:
        plt.imsave(save % 0, u)

    for i in range(nit):
        for c in range(u0.shape[2]):
            gx[..., c] = correlate2d(u[..., c], fx, 'same', 'symm')
            gy[..., c] = correlate2d(u[..., c], fy, 'same', 'symm')
        
            u[..., c] += .24 * (correlate2d(gx[..., c] - vx[..., c], bx, 'same', 'symm') + 
                                correlate2d(gy[..., c] - vy[..., c], by, 'same', 'symm'))
        
        u[u < 0] = 0
        u[u > 1] = 1

        if save and i % save_every == 0:
            plt.imsave(save % i, u)

    return u


def gdip_anisotropic(u0, vx, vy, nit=501, kappa=1e-2, diff='FB',
                            save=None, save_every=100,
                            isotropic=False, debug=False, linear=True, diff_struct=True):
    """
    Gradient domain image processing with anisotropic diffusion

    Parameters
    ----------
    u0 : ndarray (M x N x 3)
        The original image
    vx : ndarray (M x N x 3)
        The x component of the gradient to match    
    vy : ndarray (M x N x 3)
        The y component of the gradient to match    
    nit : int
        Number of iterations
    kappa : float
        anisotropy parameter
    diff : str
        finite difference method (FB, cent, Sobel, SobelFB, Feldman,
        FeldmanFB)
    isotropic : bool
        isotropic instead of anisotropi
    debug : bool
        print number of iterations every 10
    linear : bool
        compute the linearised equation (for stability)
    diff_struct : bool
        use the difference structure tensor instead of the standard structure tensor

    Returns
    -------
    im : ndarray
        The gradient processed image
    """

    # Initialize

    fx, fy, bx, by = diff_filters(diff)

    gx = np.zeros(u0.shape)
    gy = np.zeros(u0.shape)
    gxx = np.zeros(u0.shape)
    gyy = np.zeros(u0.shape)

    D11, D22, D12 = diffusion_tensor(u0, vx, vy, kappa, diff, isotropic, diff_struct)

    u = u0.copy()

    if save:
        plt.imsave(save % 0, u)

    # Iterate

    for i in range(nit):

        if (i % 10 == 0) and debug: print(i)

        if not linear:
            D11, D22, D12 = diffusion_tensor(u, vx, vy, kappa, diff, isotropic, diff_struct)

        # Anisotropic diffusion

        for c in range(3):
            gx[..., c] = correlate2d(u[..., c], fx, 'same', 'symm')
            gy[..., c] = correlate2d(u[..., c], fy, 'same', 'symm')
            gxx[..., c] = correlate2d(D11 * (gx[..., c] - vx[..., c]) +
                                      D12 * (gy[..., c] - vy[..., c]),
                                      bx, 'same', 'symm')
            gyy[..., c] = correlate2d(D12 * (gx[..., c] - vx[..., c]) +
                                      D22 * (gy[..., c] - vy[..., c]),
                                      by, 'same', 'symm')

        u += .24 * (gxx + gyy)

        u[u < 0] = 0
        u[u > 1] = 1

        # Save

        if save and i % save_every == 0:
            plt.imsave(save % i, u)

    return u
