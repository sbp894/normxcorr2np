clear;
clc;

rng(0);

A = randn(33, 200);
T = rand(33, 79);
Cout= normxcorr2(T,A);
Cout_valid= Cout(size(T,1):size(A,1), size(T,2):size(A, 2));

save('mat_nx2valid_data.mat', "A", "T","Cout_valid", "Cout")