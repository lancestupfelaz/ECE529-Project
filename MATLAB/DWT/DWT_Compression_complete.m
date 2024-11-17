clear
close all
im=imread('C:/Users/lstup.LANCE/source/repos/ECE529/ECE529-Project/Data/paris-1213603.jpg');
%% Input Image
grayScaleImg = rgb2gray(im);
InputImage_bytes = size(grayScaleImg,1) * size(grayScaleImg,2);
disp('Input image size: ' + string(InputImage_bytes/1000) + '[kb]');

%% Wavelet transform
wavelet_name = 'db1';
levels_of_decomposition = 4;
[c,s] = wavedec2(grayScaleImg, levels_of_decomposition, wavelet_name);

%% Thresholding
percentOfCoefficentsKept = 0.1; % 10%

% find the idx of the 90th percentile 
c_sorted = sort(abs(c));
c_index = floor((1 - percentOfCoefficentsKept) * length(c));
c_threshold = c_sorted(c_index+1);

keepIdxs = abs(c) > c_threshold;
c_sparse = c .* keepIdxs;

% reduce sparse array to a two-D non sparse array
j = 1;
idxs = zeros(1, sum(keepIdxs));
for i=1:length(keepIdxs)
    if (keepIdxs(i) == 1)
        idxs(1,j) = i;
        j = j + 1;
    end
end

vals = c(idxs);
%% Transmission
compressedImg_idx = uint32(idxs);
compressedImg_vals = int16(vals);

% lets quantize the compressed imag values lets round to the nearest int
figure; 
[h] = histogram(compressedImg_vals, 'BinWidth',1);


TransmittedImage_bytes = ( length(compressedImg_idx) + length(compressedImg_vals) ) * 4 ;
disp('Transmitted image size: ' + string(TransmittedImage_bytes/1000) + '[kb]');

%% Reconstruction
coefficentArray = zeros(1,length(c));
% add in transmitted coefficents
j = 1;
for idx=compressedImg_idx
    coefficentArray(idx) = compressedImg_vals(j);
    j = j + 1;
end

outputImage = uint8(waverec2(coefficentArray,s, wavelet_name));

%% plotting
figure;
subplot(1,2,1); imshow(grayScaleImg);
subplot(1,2,2); imshow(outputImage);
CompressionRatio = InputImage_bytes / TransmittedImage_bytes;

disp("Compression Ratio: " + num2str(CompressionRatio) )