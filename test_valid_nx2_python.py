
import numpy as np
from normxcorr2sp import normxcorr2,normxcorr2valid
from matlab_helpers import loadmat

mat_data= loadmat('mat_nx2valid_data.mat')
A= np.array(mat_data["A"])
T= np.array(mat_data["T"])

Cout_py= normxcorr2(T,A) 
Cout_valid_py= normxcorr2valid(T,A) 

Cout_py_mat= np.array(mat_data["Cout"])
Cout_valid_mat= np.array(mat_data["Cout_valid"])

diff_Cout = Cout_py - Cout_py_mat
max_diff_Cout = np.max(np.abs(diff_Cout))
diff_Cout_valid =  Cout_valid_py - Cout_valid_mat
max_diff_Cout_valid = np.max(np.abs(diff_Cout_valid ))

print(f"max diff: cout={diff_Cout.max()} and valid={max_diff_Cout_valid}")