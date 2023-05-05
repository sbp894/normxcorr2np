########################################################################################
# Author: Satya Parida, University of Pittsburgh, 2023 
# Based on Matlab's implementation
# Matlab normxcorr2 implementation in python 3.10.6
########################################################################################

import numpy as np
from scipy.signal import fftconvolve,convolve2d

#%% function [T, A] = ParseInputs(varargin)
#--------------------------------------------------------------------------
def ParseInputs(T,A):
    
    assert isinstance(T, np.ndarray)
    assert isinstance(A, np.ndarray)
    checkSizesTandA(T,A)
    
    # See geck 342320. If either A or T has a minimum value which is negative, we
    # need to shift the array so all values are positive to ensure numerically
    # robust results for the normalized cross-correlation.
    A = shiftData(A)
    T = shiftData(T)
    
    checkIfFlat(T);
    return (T,A)

#%% function checkSizesTandA(T,A)

def checkSizesTandA(T,A):
    if np.prod(T.shape)<2:
        raise AssertionError('Invalid template as size < 2')
        
    if np.any(np.array(T.shape)>np.array(A.shape)):
        raise AssertionError('Template must not be larger than the image in both dimensions')

#%% function checkIfFlat(T)

def checkIfFlat(T):
    if A.std==0:
        raise AssertionError('All elements in the template must not be the same')

#%% function B = shiftData(A)

def shiftData(A): 

    B = A.astype('float64')
    
    is_unsigned = np.in1d(A.dtype, ('uint8','uint16','uint32','uint64'))
    if not is_unsigned:
        min_B = np.min(B)
        
        if min_B < 0:
            B = B - min_B
    return B

#%% Function  time_conv2
# -------------------------------------------------------------------------

def time_conv2(obssize, refsize):
# These numbers were calculated for Matlab 
# May not be the appropriate numbers for Python but a good starting point

    K = 2.7e-8
    # convolution time = K*prod(obssize)*prod(refsize)
    time = K*np.prod(obssize)*np.prod(refsize)
    return time
#%% function time = time_fft2(outsize)
# -------------------------------
# Function  time_fft2
#
def time_fft2(outsize):

    # time a frequency domain convolution by timing two one-dimensional ffts
    R = outsize[0]
    S = outsize[1]
    
    # Tr = time_fft(R);
    # K_fft = Tr/(R*log(R)); 
    
    # K_fft was empirically calculated by the 2 commented-out lines above.
    K_fft = 3.3e-7; 
    Tr = K_fft*R*np.log(R)
    
    if S==R:
        Ts = Tr
    else:
        #    Ts = time_fft(S);  # uncomment to estimate explicitly
       Ts = K_fft*S*np.log(S)
        
    time = S*Tr + R*Ts;
    
    return time 

#%% function xcorr_ab = freqxcorr(a,b,outsize)
#--------------------------------------------------------------------------
# Function  freqxcorr
#
#
def freqxcorr(a,b,outsize):

    # calculate correlation in frequency domain
    Fa = np.fft.fft2(np.rot90(a,1),outsize[0],outsize[1])
    Fb = np.fft.fft2(b,outsize[0],outsize[1])
    xcorr_ab = np.fft.ifft2(Fa * Fb,'symmetric')

    return xcorr_ab

#%% Function  xcorr2_fast

def xcorr2_fast(T, A):
    T_size = np.array(T.shape)
    A_size = np.array(A.shape)
    outsize = T_size + A_size - 1
    
    # figure out when to use spatial domain vs. freq domain
    conv_time = time_conv2(T_size,A_size) # 1 conv2
    fft_time = 3*time_fft2(outsize) # 2 fft2 + 1 ifft2
    
    if conv_time < fft_time:
        cross_corr = convolve2d(np.rot90(T,1),A)
    else:
        cross_corr = freqxcorr(T,A,outsize)
        
    return cross_corr

#%% function local_sum_A = local_sum(A,m,n)
#--------------------------------------------------------------------------
# Function  local_sum
#

# We thank Eli Horn for providing this code, used with his permission,
# to speed up the calculation of local sums. The algorithm depends on
# precomputing running sums as described in "Fast Normalized
# Cross-Correlation", by J. P. Lewis, Industrial Light & Magic.

def local_sum(A,m,n):
    B = np.pad(A,((m,m,),(n,n,)))
    s = np.cumsum(B,0)
    c = s[m:-1,:] - s[0:-m-1,:]
    s = np.cumsum(c,1)
    local_sum_A = s[:,n:-1]-s[:,0:-n-1]
    return local_sum_A 

#%% NORMXCORR2 Normalized two-dimensional cross-correlation.
def normxcorr2(template, image):
    """
    Input arrays should be floating point numbers.
    :param template: N-D array, of template or filter you are using for cross-correlation.
    Must be less or equal dimensions to image.
    Length of each dimension must be less than length of image.
    :param image: N-D array
    :param mode: Options, "full", "valid", "same"
    full (Default): The output of fftconvolve is the full discrete linear convolution of the inputs. 
    Output size will be image size + 1/2 template size in each dimension.
    valid: The output consists only of those elements that do not rely on the zero-padding.
    same: The output is the same size as image, centered with respect to the ‘full’ output.
    :return: N-D array of same dimensions as image. Size depends on mode parameter.
    """
    
    [T, A] = ParseInputs(template, image)
    xcorr_TA = xcorr2_fast(T,A)
    
    (m, n) = T.shape
    mn = m*n
    
    local_sum_A = local_sum(A,m,n)
    local_sum_A2 = local_sum(A*A,m,n)
    
    # Note: diff_local_sums should be nonnegative, but may have negative
    # values due to round off errors. Below, we use max to ensure the
    # radicand is nonnegative.
    diff_local_sums = ( local_sum_A2 - (local_sum_A**2)/mn )
    denom_A = np.sqrt( np.max(diff_local_sums,0) ) 
    
    denom_T = np.sqrt(mn-1)*T.std()
    denom = denom_T*denom_A
    numerator = (xcorr_TA - local_sum_A*np.sum(T)/mn )
    
    # We know denom_T~=0 from input parsing
    # so denom is only zero where denom_A is zero, and in 
    # these locations, C is also zero.
    C = np.zeros(numerator.shape)
    
    
    tol = np.sqrt( np.eps( np.max(np.abs(denom))) )
    i_nonzero = np.find(denom > tol)
    C[i_nonzero]= numerator[i_nonzero] / denom[i_nonzero]
    
    # Another numerics backstop. If any of the coefficients are outside the
    # range [-1 1], the numerics are unstable to small variance in A or T. In
    # these cases, set C to zero to reflect undefined 0/0 condition.
    C[( np.abs(C) - 1 ) > np.sqrt(np.eps(1))] = 0
    C = np.real(C)
    
    
#%%  Debugging 
A = np.arange(20).reshape((4,5)) - 4
T = np.arange(8).reshape((2,4)) - 4
m,n = T.shape
local_sum_A = local_sum(A,m,n)
local_sum_A2 = local_sum(A*A,m,n)

[T, A] = ParseInputs(T, A)