clear
im=imread('C:/Users/lstup.LANCE/source/repos/ECE529/ECE529-Project/Data/paris-1213603.jpg');
%% Input Image
grayScaleImg = rgb2gray(im);
InputImage_bytes = size(grayScaleImg,1) * size(grayScaleImg,2);
disp('Input image size: ' + string(InputImage_bytes/1000) + '[kb]');

%% Wavelet transform
wavelet_name = 'db1';
levels_of_decomposition = 1;
[c,s] = wavedec2(grayScaleImg, levels_of_decomposition, wavelet_name);

% level 1
A1 = appcoef2(c,s,wavelet_name,1);
[H1 V1 D1]  = detcoef2('a',c,s,1);

% level 2
% A2 = appcoef2(c,s,wavelet_name,2);
% [H2 V2 D2]  = detcoef2('a',c,s,2);

% dec2 = [A2 H2; V2 D2];
dec1 = [A1 H1; V1 D1];
figure; image(dec1);