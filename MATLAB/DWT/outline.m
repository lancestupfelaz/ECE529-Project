% Load original image. 
load woman; x = X(100:200,100:200); 
nbc = size(map,1);

% Wavelet decomposition of x. 
n = 5; w = 'sym2'; [c,l] = wavedec2(x,n,w);

% Wavelet coefficients thresholding. 
thr = 20; 
keepapp = 1;
[xd,cxd,lxd,perf0,perfl2] = ...
                 wdencmp('gbl',c,l,w,n,thr,'h',keepapp);


figure; imagesc(x); title("input image");
figure; imagesc(xd); title("wavelet resconstruction");


%% Loading of an image
im=imread('C:/Users/lstup.LANCE/source/repos/ECE529/ECE529-Project/Data/paris-1213603.jpg');
% isolate channels
r = im(:,:,1);
g = im(:,:,2);
b = im(:,:,3);

[c,s] = wavedec2(r,2,'haar');
A1 = appcoef2(c,s,'haar',1);
A1img = wcodemat(A1,255,'mat',1);

figure;
imagesc(A1img)
colormap pink(255)
title('Approximation Coef. of Level 1')

% lets decompose each channel
