% scirpt to create testing dataset for normxcorr2 in python

clear;
clc;

%% copy images 
rng(1);
nImages= 50;
root_image_in_dir= 'D:\ClusterOutput_Phonemes\phoneme_cog\vIHC\level65_dBspl_clean/';
root_image_out_dir= '.\images\';
mkdir_sp(root_image_out_dir)
all_images= dir([root_image_in_dir '**' filesep '*.mat']);
images2use= randsample(all_images, nImages);

for imageVar=1:nImages
    cur_cog_fStruct= images2use(imageVar);
    cur_cog_fName_in= [cur_cog_fStruct.folder filesep cur_cog_fStruct.name];
    cur_cog_fName_out= [root_image_out_dir cur_cog_fStruct.name];

    if ~exist(cur_cog_fName_out, 'file')
        copyfile(cur_cog_fName_in, cur_cog_fName_out)
        fprintf('(%d/%d) Copied %s to \n \t \t%s\n', imageVar, nImages, cur_cog_fName_in, cur_cog_fName_out);
    else 
        fprintf('(%d/%d) %s exists\n', imageVar, nImages, cur_cog_fName_out);
    end
end
%% copy templates 
rng(1);
nTemplates= 60;
root_template_dir_in= 'D:\ClusterOutput_Phonemes\MIFanalysis_runs_phoneme\INaa_vs_OUTcallsinlist\level65_clean_fs1000Hz_norm\fragments\';
root_template_dir_out= '.\templates\';
mkdir_sp(root_template_dir_out)
all_templates= dir([root_template_dir_in '*.mat']);
templates2use= randsample(all_templates, nTemplates);

for templateVar=1:nTemplates
    cur_temp_fStruct= templates2use(templateVar);
    cur_temp_fName_in= [cur_temp_fStruct.folder filesep cur_temp_fStruct.name];
    cur_temp_fName_out= [root_template_dir_out cur_temp_fStruct.name];

    if ~exist(cur_temp_fName_out, 'file')
        copyfile(cur_temp_fName_in, cur_temp_fName_out)
        fprintf('(%d/%d) Copied %s to \n \t \t%s\n', templateVar, nTemplates, cur_temp_fName_in, cur_temp_fName_out);
    else 
        fprintf('(%d/%d) %s exists\n', templateVar, nTemplates, cur_temp_fName_out);
    end
end