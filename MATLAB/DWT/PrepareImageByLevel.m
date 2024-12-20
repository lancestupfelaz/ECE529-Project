clear
close all
im=imread('C:/Users/lstup.LANCE/source/repos/ECE529/ECE529-Project/Data/paris-1213603.jpg');
%% Input Image
grayScaleImg = rgb2gray(im);
InputImage_bytes = size(grayScaleImg,1) * size(grayScaleImg,2);
disp('Input image size: ' + string(InputImage_bytes/1000) + '[kb]');

%% Wavelet transform
wavelet_name = 'haar';
levels_of_decomposition = 2;
[c,s] = wavedec2(grayScaleImg, levels_of_decomposition, wavelet_name);
packed_output = [];

percentOfCoefficentsKept = [0.01 0.01 00.1 0.5]; % 10%
TransmittedImage_bytes = 0;

for level=1:levels_of_decomposition
    %% prepare coeficents
    A = appcoef2(c,s,wavelet_name,level);
    [H V D]  = detcoef2('a',c,s,level);

    A_flat = reshape(A, [size(A,1) * size(A,2) 1]);
    H_flat = reshape(H, [size(H,1) * size(H,2) 1]);
    V_flat = reshape(V, [size(V,1) * size(V,2) 1]);
    D_flat = reshape(D, [size(D,1) * size(D,2) 1]);
    

    c_level = [H_flat; V_flat; D_flat];

%     disp("Level " + num2str(level) + "H detailed matrix has power " + num2str(db(sum(H_flat.^2))) + "db" );
%     disp("Level " + num2str(level) + "V detailed matrix has power " + num2str(db(sum(V_flat.^2))) + "db" );
%     disp("Level " + num2str(level) + "D detailed matrix has power " + num2str(db(sum(D_flat.^2))) + "db" );

    %% Thresholding only thresholding detail coefficents

    c_sorted = sort(abs(c_level));
    c_index = floor((1 - percentOfCoefficentsKept(level)) * length(c_level));
    c_threshold = c_sorted(c_index+1);

    keepIdxs = abs(c_level) > c_threshold;
    c_sparse = c_level .* keepIdxs;

    sparseness = sum(c_sparse == 0) / length(c_sparse);
    %% Quantize to 8 bits
    dynamic_range = max(c_sparse) - min(c_sparse);
    
    CR = dynamic_range / 2^7;
    quantized_c_sparse = int8(c_sparse / CR);
    
    %figure; histogram(c_sparse);
    %figure; histogram(quantized_c_sparse);
    
    %% convert to sparse matrix
    j = 1;
    idxs = zeros(1, sum(keepIdxs));
    for i=1:length(keepIdxs)
        if (keepIdxs(i) == 1)
            idxs(1,j) = i;
            j = j + 1;
        end
    end

    c_packed = quantized_c_sparse(idxs);

    %% Hoffman encoding
    symbols = unique(c_packed);
    prob_by_symbol = zeros(1, length(symbols));
    j = 1;
    for symbol=symbols'
        prob_by_symbol(j) = sum(symbols == symbol);
        j = j + 1;
    end
    prob_by_symbol = prob_by_symbol / sum(prob_by_symbol);

    dict = huffmandict(symbols, prob_by_symbol); 
    code_hoff = huffmanenco(c_packed, dict);
    
    % measure the code length here
    disp("Level " + num2str(level) + " packed matrix is " + num2str(length(code_hoff) / 8000) + "kb" );




    
    level_compressed(level) = {c_packed};
    level_idxs(level) = {idxs};
    level_size(level) = {length(c_level)};
    level_cr  (level) = CR;
    TransmittedImage_bytes = TransmittedImage_bytes + length(c_packed) * 4 + length(level_idxs) * 4 + 4; % bytes
end

%% reconstuct levels to re-create c matrix
c_reconstucted = [];
approx_size = length(A_flat) * 4;
disp("Approximate Coefficents size: " + num2str(approx_size/1000) + "kb");
disp("Detailed Coefficents size: " + num2str(TransmittedImage_bytes/1000) + "kb");

for level=1:levels_of_decomposition
    M = zeros(1, cell2mat(level_size(level)));
    M(cell2mat(level_idxs(level))) = double(cell2mat(level_compressed(level))) .* level_cr(level);
    c_reconstucted = [M c_reconstucted];
end
c_reconstucted = [A_flat' c_reconstucted];
TransmittedImage_bytes = TransmittedImage_bytes + approx_size;

outputImage = uint8(waverec2(c_reconstucted, s, wavelet_name));

figure;
subplot(1,2,1); imshow(grayScaleImg);
subplot(1,2,2); imshow(outputImage);
CompressionRatio = InputImage_bytes / TransmittedImage_bytes;

disp("Compression Ratio: " + num2str(CompressionRatio) )
