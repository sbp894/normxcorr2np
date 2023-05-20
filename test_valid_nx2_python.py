
import numpy as np
import tensorflow as tf 
import normxcorr2np as nx2np
import normxcorr2tf as nx2tf 

# from normxcorr2np import normxcorr2,normxcorr2valid

from matlab_helpers import loadmat

mat_data= loadmat('mat_nx2valid_data.mat')
A= np.array(mat_data["A"])
T= np.array(mat_data["T"])

Cout_np= nx2np.normxcorr2(T,A) 
Cout_max_np,Cout_valid_np= nx2np.normxcorr2max(T,A) 

Cout_tf= nx2tf.normxcorr2(tf.convert_to_tensor(T),tf.convert_to_tensor(A))
Cout_max_tf,Cout_valid_tf= nx2tf.normxcorr2max(tf.convert_to_tensor(T),tf.convert_to_tensor(A)) 

Cout_np_mat= np.array(mat_data["Cout"])
Cout_valid_mat= np.array(mat_data["Cout_valid"])
Cout_max_mat= Cout_valid_mat.max()

max_diff_Cout_np = np.max(np.abs(Cout_np - Cout_np_mat))
max_diff_Cout_tf= np.max(np.abs(Cout_tf.numpy() - Cout_np_mat))

print(f"max diff: np={max_diff_Cout_np} and valid={max_diff_Cout_tf}")

print(f"Max corr vals: mat={Cout_max_mat},np={Cout_max_np},tf={Cout_max_tf}")