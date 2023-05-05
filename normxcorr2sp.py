########################################################################################
# Author: Satya Parida, University of Pittsburgh, 2023 
# Based on Matlab's implementation
# Matlab normxcorr2 implementation in python 3.10.6
########################################################################################

import numpy as np
from scipy.signal import fftconvolve

#%% Function  time_conv2
# -------------------------------------------------------------------------
#

def time_conv2(obssize, refsize):
# These numbers were calculated for Matlab 
# May not be the appropriate numbers for Python but a good starting point

    K = 2.7e-8
    # convolution time = K*prod(obssize)*prod(refsize)
    time = K*np.prod(obssize)*np.prod(refsize)
    return time

#%% Function  xcorr2_fast

def xcorr2_fast(T, A):
    T_size = np.array(T.shape)
    A_size = np.array(A.shape)
    outsize = T_size + A_size - 1
    
    # figure out when to use spatial domain vs. freq domain
    conv_time = time_conv2(T_size,A_size) # 1 conv2
    fft_time = 3*time_fft2(outsize) # 2 fft2 + 1 ifft2
    
    if (conv_time < fft_time)
        cross_corr = conv2(rot90(T,2),A)
    else
        cross_corr = freqxcorr(T,A,outsize)
    end
    
    return cross_corr


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
    xcorr_TA = xcorr2_fast(T,A)
    
    [m, n] = size(T)
    mn = m*n
    
    local_sum_A = local_sum(A,m,n)
    local_sum_A2 = local_sum(A.*A,m,n)
    
    # Note: diff_local_sums should be nonnegative, but may have negative
    # values due to round off errors. Below, we use max to ensure the
    # radicand is nonnegative.
    diff_local_sums = ( local_sum_A2 - (local_sum_A.^2)/mn )
    denom_A = sqrt( max(diff_local_sums,0) ) 
    
    denom_T = sqrt(mn-1)*std(T(:))
    denom = denom_T*denom_A
    numerator = (xcorr_TA - local_sum_A*sum(T(:))/mn )
    
    # We know denom_T~=0 from input parsing
    # so denom is only zero where denom_A is zero, and in 
    # these locations, C is also zero.
    if coder.target('MATLAB')
        C = zeros(size(numerator))
    else
        # following is needed as the 'non-symmetric' in ifft2 function
        # always gives the complex results (as xcorr_TA is complex)
     C = zeros(size(numerator),'like',xcorr_TA)
    end
    
    tol = sqrt( eps( max(abs(denom(:)))) )
    i_nonzero = find(denom > tol)
    C(i_nonzero) = numerator(i_nonzero) ./ denom(i_nonzero)
    
    # Another numerics backstop. If any of the coefficients are outside the
    # range [-1 1], the numerics are unstable to small variance in A or T. In
    # these cases, set C to zero to reflect undefined 0/0 condition.
    C( ( abs(C) - 1 ) > sqrt(eps(1)) ) = 0
    if ~coder.target('MATLAB')
        C = real(C)
    end