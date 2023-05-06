# -*- coding: utf-8 -*-
"""
Created on Fri May  5 11:53:32 2023

@author: spsat
"""

import os 
import numpy as np 
from normxcorr2sp import normxcorr2
from matlab_helpers import loadmat 
from scipy.io import savemat  # matlab dsearchn equivalent 
import matplotlib.pyplot as plt

#%% Read image filenames 
image_dir= 'images/'
all_images= []
for img_file in os.listdir(image_dir):
    if img_file.endswith('.mat'):
        cur_img_file = os.path.join(image_dir, img_file)
        print(f"Working on {cur_img_file}")
        all_images.append(cur_img_file)

#%% Read image templates 
tempalte_dir= 'templates/'
all_templates= []
for template_file in os.listdir(tempalte_dir):
    if template_file.endswith('.mat'):
        cur_template_file = os.path.join(tempalte_dir, template_file)
        print(f"Working on {cur_template_file}")
        all_templates.append(cur_template_file)
        
#%% Compute and save normxcorr2 values like in Matlab 
fNmae2save_python= 'python_normxcorr2_output.mat';

if not os.path.isfile(fNmae2save_python):
    corr_energy_python= np.zeros((len(all_images), len(all_templates)))
    
    for img_index,img_file in enumerate(all_images):
        img_struct_data= loadmat(img_file)
        img_struct_data= img_struct_data['cochleogram']
        
        for temp_index,template_file in enumerate(all_templates):
            template_struct_data= loadmat(template_file)
            template_struct_data= template_struct_data['frag']
            
            img_freq_start= np.argmin(np.abs(img_struct_data['centerfreq']-template_struct_data['freqlower']))
            img_freq_end= np.argmin(np.abs(img_struct_data['centerfreq']-template_struct_data['frequpper']))
            
            cur_image_data= img_struct_data['meanrate'][img_freq_start:img_freq_end+1,:]
            cur_template_data= template_struct_data['data']
            
            len_diff= cur_template_data.shape[1] - cur_image_data.shape[1]
            if len_diff>0:
                cur_image_data= np.concatenate((cur_image_data, np.zeros((cur_template_data.shape[0],len_diff))), axis=1)
            
            cur_corr= normxcorr2(cur_template_data, cur_image_data)
            corr_energy_python[img_index,temp_index] = np.sum(cur_corr**2)
    
    data2save= {'corr_energy':corr_energy_python}
    savemat(fNmae2save_python, data2save)
else:
    corr_energy_python = loadmat(fNmae2save_python)
    corr_energy_python = corr_energy_python['corr_energy']
    
#%% Compare with Matlab 
fNmae2save_matlab= 'matlab_normxcorr2_output.mat'
corr_energy_matlab= loadmat(fNmae2save_matlab)
corr_energy_matlab= corr_energy_matlab['corr_energy']

plt.figure(num=1, figsize=(8,4))
plt.clf()

plt.subplot(1,2,1)
plt.scatter(corr_energy_python.flatten(), corr_energy_matlab.flatten())
plt.xlabel('Python')
plt.ylabel('Matlab')
plt.title('normxcorr2 comparison')

plt.subplot(1,2,2)
plt.hist( (corr_energy_python.flatten()-corr_energy_matlab.flatten()) /  corr_energy_matlab.flatten() * 100)
plt.xlabel('%age error')
plt.ylabel('Count')
plt.title('error histogram')

plt.tight_layout()
plt.savefig('comaprison_fig.png',dpi=300)