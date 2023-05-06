clear;
clc;

fNmae2save= 'matlab_normxcorr2_output.mat';
force_redo= 1;

if ~exist(fNmae2save, 'file') | force_redo

    image_dir= ['images' filesep];
    template_dir= ['templates' filesep];

    all_images= dir([image_dir '*.mat']);
    all_templates= dir([template_dir '*.mat']);


    corr_energy= nan(length(all_images), length(all_templates));

    for imgVar=1:length(all_images)
        cur_image_fStruct= all_images(imgVar);
        cur_image_fName= [cur_image_fStruct.folder filesep cur_image_fStruct.name];
        cur_image_struct_data= load(cur_image_fName);
        cur_image_struct_data= cur_image_struct_data.cochleogram;


        for tempVar=1:length(all_templates)
            cur_temp_fStruct= all_templates(tempVar);
            cur_temp_fName= [cur_temp_fStruct.folder filesep cur_temp_fStruct.name];
            cur_temp_struct_data= load(cur_temp_fName);
            cur_temp_struct_data= cur_temp_struct_data.frag;

            img_freq_ind_start = dsearchn(cur_image_struct_data.centerfreq(:), cur_temp_struct_data.freqlower);
            img_freq_ind_end = dsearchn(cur_image_struct_data.centerfreq(:), cur_temp_struct_data.frequpper);

            cur_image = cur_image_struct_data.meanrate(img_freq_ind_start:img_freq_ind_end, :);
            cur_template = cur_temp_struct_data.data;

            if size(cur_image,2) < size(cur_template,2)
                len_diff= size(cur_template,2) - size(cur_image,2);
                cur_image= [cur_image, zeros(size(cur_image,1), len_diff)];
            end

            cur_corr = normxcorr2(cur_template, cur_image);
            corr_energy(imgVar, tempVar)= sum(cur_corr(:).^2);
        end
    end

%     save(fNmae2save, 'corr_energy')
end