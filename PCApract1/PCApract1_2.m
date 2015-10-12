% The PCA practise on image comprassion
% Shijian Fan
% 08/11/2015
close all;
clear all;
clc;
img = imread('lena_gray_512.tiff'); % load image into MATLAB
img=double(img); % convert to double precision
figure,imshow(img); % display image
title('The Original Lena Img.');
[m n]=size(img);
mn = mean(img,2); % compute row mean
X = img - repmat(mn,1,n); % subtract row mean to obtain X
%Z=1/sqrt(n-1)*X'; % create matrix, Z
%covZ=Z'*Z; % covariance matrix of Z
%% Singular value decomposition
PCs=100;
tic;
[U,S,V] = svd(X);
toc;
variances=diag(S).*diag(S); % compute variances
figure,bar(variances(1:PCs)) % scree plot of variances
%% Extract principal components

UU = zeros(size(U));
UU(:,1:PCs)=U(:,1:PCs);
Y=UU'*X; % project data onto PCs
ratio=256/(2*PCs+1); % compression ratio
XX=UU*Y; % convert back to original basis
XX=XX+repmat(mn,1,n); % add the row means back on
figure,imshow(XX); % display results
title('The Compressed Lena Img.');
