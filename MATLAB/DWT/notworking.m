clear
close all
im=imread('C:/Users/lstup.LANCE/source/repos/ECE529/ECE529-Project/Data/paris-1213603.jpg');
%% Input Image
grayScaleImg = rgb2gray(im);
InputImage_bytes = size(grayScaleImg,1) * size(grayScaleImg,2);
disp('Input image size: ' + string(InputImage_bytes/1000) + '[kb]');

%% Wavelet transform
wavelet_name = 'db1';
levels_of_decomposition = 2;
[c,s] = wavedec2(grayScaleImg, levels_of_decomposition, wavelet_name);

%% Thresholding
percentOfCoefficentsKept = 0.02; % 10%

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
% lets compress the image idexes too
[values, starts, lengths] = runLengthEncode(c_sparse);

%% Quantize to 8 bits
dynamic_range = max(vals) - min(vals);
    
CR = dynamic_range / 2^7;
quantized_c_sparse = int8(vals / CR);

compressedImg_vals = quantized_c_sparse;

% lets quantize the compressed imag values lets round to the nearest int
symbol = unique(compressedImg_vals);
prob_by_symbol = zeros(1, length(symbol));

idx = 1;
for sym=symbol
    prob_by_symbol(1,idx) = sum(compressedImg_vals == sym);
    idx = idx + 1;
end
prob_by_symbol = prob_by_symbol / sum(prob_by_symbol);
hold on;
figure; plot(symbol, prob_by_symbol); title('probablity by bin');

%%%%%%%%%%%%%%%%%%%%%%%%%%
dict = huffmandict(symbol, prob_by_symbol); 

code_hoff = huffmanenco(compressedImg_vals,dict);
% measure data size here
TransmittedImage_bytes = ( length(compressedImg_idx) * 4 + length(code_hoff)/8 ) ;
disp('Transmitted image size: ' + string(TransmittedImage_bytes/1000) + '[kb]');

%%%%%



compressedImg_vals = huffmandeco(code_hoff,dict);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Reconstruction
coefficentArray = zeros(1,length(c));
% add in transmitted coefficents
j = 1;
for idx=compressedImg_idx
    coefficentArray(idx) = double(compressedImg_vals(j)) * CR;
    j = j + 1;
end

outputImage = uint8(waverec2(coefficentArray,s, wavelet_name));

%% plotting
figure;
subplot(1,2,1); imshow(grayScaleImg);
subplot(1,2,2); imshow(outputImage);
CompressionRatio = InputImage_bytes / TransmittedImage_bytes;

disp("Compression Ratio: " + num2str(CompressionRatio) )

function [values, starts, lengths] = runLengthEncode(array)
    % Find runs of non-zero elements
    non_zero_indices = find(array ~= 0);
    values = array(non_zero_indices);
    
    % Calculate run lengths
    starts = [non_zero_indices(1) non_zero_indices(find(diff(non_zero_indices) > 1) + 1)];
    ends = [non_zero_indices(find(diff(non_zero_indices) > 1)) non_zero_indices(end)];
    lengths = ends - starts + 1;
end