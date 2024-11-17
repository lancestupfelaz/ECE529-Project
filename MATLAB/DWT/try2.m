close all
clear
%% Loading of an image
im=imread('C:/Users/lstup.LANCE/source/repos/ECE529/ECE529-Project/Data/paris-1213603.jpg');
% isolate channels
r = im(:,:,1);
g = im(:,:,2);
b = im(:,:,3);
InputImage_bytes = size(r,1) * size(r,2);
disp('Input image size: ' + string(InputImage_bytes/1000) + '[kb]');
figure;
imagesc(r)
colormap pink(255)
title('Input Image')


%% wavelet transform
[c,s] = wavedec2(r,1,'haar');



%% Quantization
approximateCoefficent = c;
Quantized_signal = wcodemat(approximateCoefficent,1000,'mat',1);
Quantized_signal_flat = reshape(Quantized_signal, [1 size(Quantized_signal,1)*size(Quantized_signal,2)]);

%% Thresholding
threshold = 0.1; % take only 10% largest coefficents
sorted_signal = sort(Quantized_signal_flat);

startIdx = (1-threshold)*length(sorted_signal);
thresholdValue = sorted_signal(floor(startIdx));

keep_idxs = Quantized_signal_flat > thresholdValue;

Quantized_signal_flat = Quantized_signal_flat .* keep_idxs;

%% entropy encoding 
figure;
h = histogram(Quantized_signal_flat,"BinWidth",1.0001);
counts = double(h.Values);

% Arithmatic code
counts = nonzeros(counts);
compressedImage = arithenco(Quantized_signal_flat, counts);


%% compressed image
compressedSize_bytes = size(compressedImage,2) / 8;
disp('Compressed Image Size ' + string(compressedSize_bytes/1000) + '[kb]');


%% entropy decoding
dseq = arithdeco(compressedImage,counts,length(compressedImage));
quantizedImageResult = uint8(dseq);


quantizedImageResult = uint8(approximateCoefficent);
%% dequantization
% for non linear quantizations
% nothing to do here :)


%% Inverse wavelet transform
xrec = waverec2(quantizedImageResult,s,'haar');

%% image reconstruction
figure;
imagesc(xrec)
colormap pink(255)
title('Approximation Coef. of Level 1')

%% Compression Ratio
CompressionRatio = InputImage_bytes / compressedSize_bytes

