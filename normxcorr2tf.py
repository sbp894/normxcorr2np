########################################################################################
# Author: Satya Parida, University of Pittsburgh, 2023 
# Based on Matlab's implementation
# Matlab normxcorr2 implementation in python 3.10.6 using tf functions 
########################################################################################

import numpy as np
import tensorflow as tf 
is_debugging= True

#%% function [T, A] = ParseInputs(varargin)
#--------------------------------------------------------------------------
def ParseInputs(T,A):

    assert isinstance(T, tf.Tensor)
    assert isinstance(A, tf.Tensor)
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
    if tf.math.reduce_prod(T.shape)<2:
        raise AssertionError('Invalid template as size < 2')
        
    if T.shape.as_list() > A.shape.as_list():
        raise AssertionError('Template must not be larger than the image in both dimensions')

#%% function checkIfFlat(T)
def checkIfFlat(T):
    # Note tf.math.reduce_std has a ddof = 1 (instead of 0 as in np.std)
    # here, it does not matter
    if tf.math.reduce_std(tf.cast(T, dtype=tf.float64))==0: 
        raise AssertionError('All elements in the template must not be the same')

#%% function B = shiftData(A)
def shiftData(A):
    B = tf.cast(A, dtype=tf.float64)
    
    is_unsigned =A.dtype in ('tf.uint8','tf.uint16','tf.uint32','tf.uint64')
    if not is_unsigned:
        min_B = tf.reduce_min(B)
        
        if min_B < 0:
            B = B - min_B
    return B

#%% Function  time_conv2
# -------------------------------------------------------------------------
def time_conv2(obssize, refsize):
# These numbers were calculated for Matlab 
# May not be the appropriate numbers for Python but a good starting point

    K = tf.constant(2.7e-8, dtype=tf.float64)
    obssize= tf.cast(obssize, dtype=tf.float64)
    refsize= tf.cast(refsize, dtype=tf.float64)
    time = K*tf.math.reduce_prod(obssize)*tf.math.reduce_prod(refsize)
    return time
#%% function time = time_fft2(outsize)
# -------------------------------
# Function  time_fft2
#
def time_fft2(outsize):
    # time a frequency domain convolution by timing two one-dimensional ffts
    R = tf.cast(outsize[0], dtype=tf.float64)
    S = tf.cast(outsize[1], dtype=tf.float64)
    
    # Tr = time_fft(R);
    # K_fft = Tr/(R*log(R)); 
    
    # K_fft was empirically calculated by the 2 commented-out lines above.
    K_fft = tf.constant(3.3e-7, dtype=tf.float64)
    Tr = K_fft*R*tf.math.log(R)
    
    if S==R:
        Ts = Tr
    else:
        #    Ts = time_fft(S);  # uncomment to estimate explicitly
       Ts = K_fft*S*tf.math.log(S)
        
    time = S*Tr + R*Ts;
    
    return time 

#%% equivalent to np.rot90()
def my_tf_rot90(A,count):
    if A.ndim==2:
        for rot_num in range(count):
            A = tf.squeeze(tf.image.rot90(A[:,:,tf.newaxis]))
    elif A.ndim > 2:
        assert False, 'not yet implemented'
        for rot_num in range(count):
            A = tf.image.rot90(A)

    return A 

#%% function xcorr_ab = freqxcorr(a,b,outsize)
#--------------------------------------------------------------------------
# Function  freqxcorr
#
def freqxcorr(a,b,outsize):

    # calculate correlation in frequency domain
    a_rotx2= my_tf_rot90(a,2)
    # becasue tf.signal.fft2d does not take nfft (outsize here) as an input, have to zero-pad before hand 
    a_ypad= [tf.constant(0), (outsize - a.shape)[0]]
    a_xpad= [tf.constant(0), (outsize - a.shape)[1]]
    a_rotx2_padded= tf.pad(a_rotx2, [a_ypad,a_xpad])
    Fa = tf.signal.fft2d(tf.cast(a_rotx2_padded, tf.complex128))
    
    b_ypad= [tf.constant(0), (outsize - b.shape)[0]]
    b_xpad= [tf.constant(0), (outsize - b.shape)[1]]
    b_padded= tf.pad(b, [b_ypad,b_xpad]) # note: b is not rotated 
    Fb = tf.signal.fft2d(tf.cast(b_padded, tf.complex128))
    xcorr_ab = tf.signal.ifft2d(Fa * Fb)

    return xcorr_ab

#%% Function  xcorr2_fast
def xcorr2_fast(T, A):
    T_size = T.shape
    A_size = A.shape
    outsize = tf.math.add(T_size, A_size) - 1
    
    # always do freqxcorr
    cross_corr = freqxcorr(T,A,outsize)
    
    # figure out when to use spatial domain vs. freq domain
    # conv_time = time_conv2(T_size,A_size) # 1 conv2
    # fft_time = 3*time_fft2(outsize) # 2 fft2 + 1 ifft2
    
    # if conv_time < fft_time:
    #     cross_corr = convolve2d(np.rot90(T,2),A)
    # else:
    #     cross_corr = freqxcorr(T,A,outsize)
        
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
    B = tf.pad(A,((m,m,),(n,n,)))
    s = tf.math.cumsum(B,0)
    c = s[m:-1,:] - s[0:-m-1,:]
    s = tf.math.cumsum(c,axis=1)
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
    diff_local_sums= ( local_sum_A2 - (local_sum_A**2)/mn)    
    
    denom_A= tf.clip_by_value(diff_local_sums, clip_value_min=0,clip_value_max=np.inf)
    denom_A= tf.math.sqrt(denom_A)
    ddof1_factor= tf.math.sqrt(tf.size(T)/(tf.size(T)-1)) #factor to unbias std estimate w/ numpy call (ddof=1)
    denom_T = tf.math.sqrt(tf.cast(mn-1,dtype=tf.float64))*tf.math.reduce_std(T)*ddof1_factor
    denom = denom_T*denom_A
    if not xcorr_TA.dtype==(local_sum_A*tf.reduce_sum(T)/mn).dtype:
        numerator = xcorr_TA - tf.cast((local_sum_A*tf.reduce_sum(T)/mn), dtype=xcorr_TA.dtype)
    else:
        numerator = xcorr_TA - local_sum_A*tf.reduce_sum(T)/mn
    
    # We know denom_T~=0 from input parsing
    # so denom is only zero where denom_A is zero, and in 
    # these locations, C is also zero.
    C = tf.zeros(numerator.shape, dtype=xcorr_TA.dtype)
    tol = tf.constant(6e-8, dtype=tf.float64)
    i_nonzero = tf.math.greater(denom, tol)

    ratio_update_index= tf.where(i_nonzero)
    if not xcorr_TA.dtype==(local_sum_A*tf.reduce_sum(T)/mn).dtype:
        ratio_update_values= numerator[i_nonzero] / tf.cast(denom[i_nonzero], dtype=numerator.dtype)
    else:
        ratio_update_values= numerator[i_nonzero] / denom[i_nonzero]
            

    C= tf.tensor_scatter_nd_update(C, ratio_update_index, ratio_update_values)
    
    # Another numerics backstop. If any of the coefficients are outside the
    # range [-1 1], the numerics are unstable to small variance in A or T. In
    # these cases, set C to zero to reflect undefined 0/0 condition.
    np_spacing1= tf.constant(2.220446049250313e-16, dtype=tf.float64) # = np.spacing(1), what matlab uses
    zero_update_logical= tf.cast((tf.math.abs(C) - 1), dtype=tf.float64) > tf.math.sqrt(np_spacing1)
    if tf.math.reduce_any(zero_update_logical):
        zero_update_index= tf.where(zero_update_logical)
        zero_update_values= tf.zeros(zero_update_index.shape[0], dtype=tf.float64)
        C= tf.tensor_scatter_nd_update(C, zero_update_index, zero_update_values)
    
    C = tf.math.real(C)
    return C
    
#%%  Verified 
def normxcorr2max_batch(template, batch_image):
    print(f"batch_image.shape={batch_image.shape}")
    assert len(batch_image.shape)==3 , "input should have 3 dims (batch,cf,time)"
    
    max_ccf = []
    for image in batch_image:
        if is_debugging:
            print(f"image.shape={image.shape},dtype={image.dtype},template.shape={template.shape}")
        ccf_out= normxcorr2(template, image)
        ccf_out= ccf_out[template.shape[0]-1:(image.shape[0]), template.shape[1]-1:(image.shape[1])]
        max_ccf.append(ccf_out.max())
        # print(f"max={max_ccf}")
    
    return max_ccf

#%%  Function for max of normxcorr2max
def normxcorr2max(template, image):
    ccf_out= normxcorr2(template, image)
    begin_slice= [template.shape[0]-1, template.shape[1]-1]
    size_slice= [image.shape[0] - (template.shape[0]-1), image.shape[1] - (template.shape[1]-1)]
    ccf_out= tf.slice(ccf_out, begin=begin_slice, size=size_slice)
    max_ccf = tf.reduce_max(ccf_out)
    return max_ccf,ccf_out

#%% np_spacing
def np_spacing(x):
    return np.spacing(x)

#%%  Verify 
do_verify = False 
do_verify_ffts = False
if do_verify :
    A = tf.convert_to_tensor(np.arange(40).reshape((4,10)) - 4, dtype=tf.float64)
    T = tf.convert_to_tensor(np.array([[16,2,3,13], [5,11,10,8], [9,7,6,12], [4,14,15,1]]), dtype=tf.float64)

    # A = tf.reshape(tf.range(20), (4,5)) - 4
    # T = tf.reshape(tf.range(8), (2,4)) - 4 
    [T, A] = ParseInputs(T, A)
    tc_val= time_conv2(T.shape,A.shape)
    outsize = tf.math.add(T.shape,A.shape) - 1
    fft_time = 3*time_fft2(outsize)
    xcorr_TA = xcorr2_fast(T,A)

    if do_verify_ffts:
        a= A
        b= T
       
        # tf 
        a_rotx2= my_tf_rot90(a,2)
        # becasue tf.signal.fft2d does not take nfft (outsize here) as an input, have to zero-pad before hand 
        a_ypad= [tf.constant(0), (outsize - a.shape)[0]]
        a_xpad= [tf.constant(0), (outsize - a.shape)[1]]
        a_rotx2_padded= tf.pad(a_rotx2, [a_ypad,a_xpad])
        Fa = tf.signal.fft2d(tf.cast(a_rotx2_padded, tf.complex64))
        
        b_ypad= [tf.constant(0), (outsize - b.shape)[0]]
        b_xpad= [tf.constant(0), (outsize - b.shape)[1]]
        b_padded= tf.pad(b, [b_ypad,b_xpad]) # note: b is not rotated 
        Fb = tf.signal.fft2d(tf.cast(b_padded, tf.complex64))
        xcorr_ab = tf.signal.ifft2d(Fa * Fb)
        
        # numpy 
        Fa_np = np.fft.fft2(np.rot90(a,2),outsize)
        Fb_np = np.fft.fft2(b,outsize)
        xcorr_ab_np = np.fft.ifft2(Fa_np * Fb_np)
        
        # compare 
        np.allclose(Fa, Fa_np)
        np.allclose(Fb, Fb_np)
        np.allclose(xcorr_ab, xcorr_ab_np)
        np.max(np.abs(xcorr_ab.numpy() - xcorr_ab_np))/np.std(xcorr_ab_np)

    m,n = T.shape
    mn = m*n
    local_sum_A = local_sum(A,m,n)
    local_sum_A2 = local_sum(A*A,m,n)
    diff_local_sums = ( local_sum_A2 - (local_sum_A**2)/mn )
    denom_A= tf.clip_by_value(diff_local_sums, clip_value_min=0,clip_value_max=np.inf)
    denom_A= tf.math.sqrt(denom_A)
    ddof1_factor= tf.math.sqrt(tf.size(T)/(tf.size(T)-1)) #factor to unbias std estimate w/ numpy call (ddof=1)
    denom_T = tf.math.sqrt(tf.cast(mn-1,dtype=tf.float64))*tf.math.reduce_std(T)*ddof1_factor
    denom = denom_T*denom_A
    numerator = (xcorr_TA - local_sum_A*np.sum(T)/mn)
    
    C = tf.zeros(numerator.shape, dtype=tf.float64)
    tol = tf.constant(6e-8, dtype=tf.float64) # this ~ np.sqrt(np.spacing(2^16))
    i_nonzero = tf.math.greater(denom, tol)
    
    ratio_update_index= tf.where(i_nonzero)
    ratio_update_values= numerator[i_nonzero] / denom[i_nonzero]
    C= tf.tensor_scatter_nd_update(C, ratio_update_index, ratio_update_values)
    
    np_spacing1= tf.constant(2.220446049250313e-16, dtype=tf.float64) # = np.spacing(1)
    zero_update_logical= ( tf.math.abs(C) - 1 ) > tf.math.sqrt(np_spacing1)
    if tf.math.reduce_any(zero_update_logical):
        zero_update_index= tf.where(zero_update_logical)
        zero_update_values= tf.zeros(zero_update_index.shape[0], dtype=tf.float64)
        C= tf.tensor_scatter_nd_update(C, zero_update_index, zero_update_values)
    
    C = tf.math.real(C)
        
do_verify_valid = True 
if do_verify_valid:
    A = tf.convert_to_tensor(np.arange(40).reshape((4,10)) - 4, dtype=tf.float64)
    T = tf.convert_to_tensor(np.array([[16,2,3,13], [5,11,10,8], [9,7,6,12], [4,14,15,1]]), dtype=tf.float64)
    C = normxcorr2(T, A)
    Cvalid= normxcorr2max(T, A)
    