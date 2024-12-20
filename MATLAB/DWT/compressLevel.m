clear
close all
im=imread('C:/Users/lstup.LANCE/source/repos/ECE529/ECE529-Project/Data/paris-1213603.jpg');
%% Input Image
grayScaleImg = rgb2gray(im);
InputImage_bytes = size(grayScaleImg,1) * size(grayScaleImg,2);
disp('Input image size: ' + string(InputImage_bytes/1000) + '[kb]');

%% Wavelet transform
wavelet_name = 'db1';
levels_of_decomposition = 2
[c,s] = wavedec2(grayScaleImg, levels_of_decomposition, wavelet_name);

% level 1
A1 = appcoef2(c,s,wavelet_name,1);
[H1 V1 D1]  = detcoef2('a',c,s,1);

A1_flat = reshape(A1, [size(A1,1) * size(A1,2) 1]);
H1_flat = reshape(H1, [size(H1,1) * size(H1,2) 1]);
V1_flat = reshape(V1, [size(V1,1) * size(V1,2) 1]);
D1_flat = reshape(D1, [size(D1,1) * size(D1,2) 1]);

% level 2
A2 = appcoef2(c,s,wavelet_name,2);
[H2 V2 D2]  = detcoef2('a',c,s,2);

A2_flat = reshape(A2, [size(A2,1) * size(A2,2) 1]);
H2_flat = reshape(H2, [size(H2,1) * size(H2,2) 1]);
V2_flat = reshape(V2, [size(V2,1) * size(V2,2) 1]);
D2_flat = reshape(D2, [size(D2,1) * size(D2,2) 1]);

level2_index = 1;
level1_index = length(A2_flat) + length(H2_flat) + length(V2_flat) + length(D2_flat);

c = [A2_flat; H2_flat; V2_flat; D2_flat; H1_flat; V1_flat; D1_flat]';

%% Thresholding
percentOfCoefficentsKept = 0.1; % 10%

% find the idx of the 90th percentile 
c_sorted = sort(abs(c));
c_index = floor((1 - percentOfCoefficentsKept) * length(c));
c_threshold = c_sorted(c_index+1);



keepIdxs = abs(c) > c_threshold;
c_sparse = c .* keepIdxs;

%% figure out how sparse the matrcies are


A1 = appcoef2(c_sparse,s,wavelet_name,1);
[H1 V1 D1]  = detcoef2('a',c_sparse,s,1);

A1_flat = reshape(A1, [size(A1,1) * size(A1,2) 1]);
H1_flat = reshape(H1, [size(H1,1) * size(H1,2) 1]);
V1_flat = reshape(V1, [size(V1,1) * size(V1,2) 1]);
D1_flat = reshape(D1, [size(D1,1) * size(D1,2) 1]);

% level 2
A2 = appcoef2(c_sparse,s,wavelet_name,2);
[H2 V2 D2]  = detcoef2('a',c_sparse,s,2);

A2_flat = reshape(A2, [size(A2,1) * size(A2,2) 1]);
H2_flat = reshape(H2, [size(H2,1) * size(H2,2) 1]);
V2_flat = reshape(V2, [size(V2,1) * size(V2,2) 1]);
D2_flat = reshape(D2, [size(D2,1) * size(D2,2) 1]);

dec2 = [A2 H2; V2 D2];
dec1 = [imresize(dec2,size(H1)) H1; V1 D1];
image(dec1);

sparse_1 = sum(c_sparse(level1_index:length(c)) == 0) / length(c_sparse(level1_index:length(c))) % 90 sparse
sparse_2 = sum(c_sparse(level2_index:level1_index) == 0) / length(c_sparse(level2_index:level1_index)) % 90 sparse






%%


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

compressedImg_vals = int16(vals);

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
TransmittedImage_bytes = ( length(compressedImg_idx)  * 4 + length(code_hoff)/8 ) ;
disp('Transmitted image size: ' + string(TransmittedImage_bytes/1000) + '[kb]');

%%%%%



compressedImg_vals = huffmandeco(code_hoff,dict);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


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

% function [values, starts, lengths] = runLengthEncode(array)
%     % Find runs of non-zero elements
%     non_zero_indices = find(array ~= 0);
%     values = array(non_zero_indices);
%     
%     % Calculate run lengths
%     starts = [non_zero_indices(1) non_zero_indices(find(diff(non_zero_indices) > 1) + 1)];
%     ends = [non_zero_indices(find(diff(non_zero_indices) > 1)) non_zero_indices(end)];
%     lengths = ends - starts + 1;
% end