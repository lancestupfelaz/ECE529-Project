%% Initialize
clear all;close all; clc;
% Read an image
% standard
image = imread('peppers.png');
% non multiple of 8
%image = imread('\\devshares\Profiles\dpreissler\My Documents\My Pictures\ASR11.PNG');
% 4k
image = imread('\\devshares\Profiles\dpreissler\My Documents\My Pictures\4kwallpaper.jpg');
% analyze format and shape of the file as a matrix
image = imread('\\devshares\Profiles\dpreissler\My Documents\My Pictures\IRimage.jpg');

whos image;
needspadding = image;
image = pad_image_to_multiple_of_8(needspadding);

% Display the original and padded images
% figure;
% subplot(1, 2, 1), imshow(needspadding), title('Original Image');
% subplot(1, 2, 2), imshow(image), title('Padded Image');
%if image isn't multiple of 8 duplicate edge pixels for coherency
% for now just truncate
quality = 98; %quality for DCT quantization table


%% DCT
% DCTencode
R = image(:,:,1);
G = image(:,:,2);
B = image(:,:,3);

% Colorspace conversion, RGB to lumanince and chrominance
YCBCR = rgb2ycbcr(image);
Y = YCBCR(:,:,1);
Cb = YCBCR(:,:,2);
Cr = YCBCR(:,:,3);

% downsample chrominance channels
Crmean = zeros(size(Cr,1)/4,size(Cr,2)/4,'uint8');
Cbmean = zeros(size(Cr,1)/4,size(Cr,2)/4,'uint8');
for m = 1:4:size(Cr,1) %m is rows
    for n = 1:4:size(Cr,2) %n is columns
        indexRow = (m-1)/4 + 1;
        indexCol = (n-1)/4 + 1;
        Crmean(indexRow,indexCol) = mean(Cr(m:m+3,n:n+3),'all');
        Cbmean(indexRow,indexCol) = mean(Cb(m:m+3,n:n+3),'all');
    end
end


% perform the forward DCT II algorithm on all channel Y, Cb, Cr
Crmean = cast(Crmean,'int16');
Cbmean = cast(Cbmean,'int16');
Crmean = Crmean - 129;
Cbmean = Cbmean - 129;
% Assuming Crmean and Cbmean are already defined
% Allocate storage for DCT coefficients
DCToutCr = zeros(size(Crmean));
DCToutCb = zeros(size(Cbmean));

% Perform DCT on 8x8 blocks and store the results
for m = 1:8:size(Crmean,1)
    for n = 1:8:size(Crmean,2)
        blockCr = Crmean(m:m+7, n:n+7);
        DCToutCr(m:m+7, n:n+7) = maxsforwardDCT(blockCr);

        blockCb = Cbmean(m:m+7, n:n+7);
        DCToutCb(m:m+7, n:n+7) = maxsforwardDCT(blockCb);
    end
end

% Define a standard JPEG quantization table for chrominance
chromqtable = [
    17 18 24 47 99 99 99 99;
    18 21 26 66 99 99 99 99;
    24 26 56 99 99 99 99 99;
    47 66 99 99 99 99 99 99;
    99 99 99 99 99 99 99 99;
    99 99 99 99 99 99 99 99;
    99 99 99 99 99 99 99 99;
    99 99 99 99 99 99 99 99;
];

% Quantize the DCT coefficients using the quantization table
for m = 1:8:size(DCToutCr,1)
    for n = 1:8:size(DCToutCr,2)
        DCToutCr(m:m+7, n:n+7) = round(DCToutCr(m:m+7, n:n+7) ./ chromqtable);
        DCToutCb(m:m+7, n:n+7) = round(DCToutCb(m:m+7, n:n+7) ./ chromqtable);
    end
end

nnz(DCToutCr)/(size(DCToutCr,1) * size(DCToutCr,2))
numbcoeffs = sum(DCToutCr > abs(2),'all');

mean(DCToutCr > abs(2),'all')
% Create a logical mask for values greater than 1 (ignoring sign)
mask = abs(DCToutCr) > 2;

% Apply the mask to the DCT coefficients
filtered_values = DCToutCr .* mask;

% Remove zero values resulting from the masking process
filtered_values = filtered_values(filtered_values ~= 0);

% Plot the histogram
hist(filtered_values);
title('Histogram of DCT Coefficients (|value| > 1)');
xlabel('DCT Coefficient Value');
ylabel('Frequency');


% DCTdecode
% perform inverse DCT II or DCTIII algorithm for all channels Y, Cb, Cr
% Allocate storage for inverse DCT coefficients
IDCToutCr = zeros(size(Crmean));
IDCToutCb = zeros(size(Cbmean));
% De-Quantization
for m = 1:8:size(DCToutCr,1)
    for n = 1:8:size(DCToutCr,2)
        DCToutCr(m:m+7, n:n+7) = DCToutCr(m:m+7, n:n+7) .* chromqtable;
        DCToutCb(m:m+7, n:n+7) = DCToutCb(m:m+7, n:n+7) .* chromqtable;
    end
end

% Perform Inverse DCT on 8x8 blocks and store the results
for m = 1:8:size(DCToutCr,1)
    for n = 1:8:size(DCToutCr,2)
        blockCr = maxsbackwardDCT(DCToutCr(m:m+7, n:n+7));
        IDCToutCr(m:m+7, n:n+7) = blockCr;
        
        blockCb = maxsbackwardDCT(DCToutCb(m:m+7, n:n+7));
        IDCToutCb(m:m+7, n:n+7) = blockCb;
    end
end

% Now IDCToutCr and IDCToutCb contain the decoded and de-quantized data


IDCToutCr = IDCToutCr + 129;
IDCToutCb = IDCToutCb + 129;

IDCToutCr = cast(IDCToutCr, 'uint8');
IDCToutCb = cast(IDCToutCb, 'uint8');
% upsample
Crup = zeros(size(Cr,1),size(Cr,2),'uint8');
Cbup = zeros(size(Cr,1),size(Cb,2),'uint8');
for m = 1:size(Crmean,1)
    for n = 1:size(Crmean,2)
        Crup((m-1)*4 + 1:(m-1)*4 + 4,(n-1)*4 + 1:(n-1)*4+4) = IDCToutCr(m,n);
        Cbup((m-1)*4 + 1:(m-1)*4 + 4,(n-1)*4 + 1:(n-1)*4+4) = IDCToutCb(m,n);
    end
end


%Back to RGB from YCrCb
imageout = ycbcr2rgb(cat(3,Y,Cbup,Crup));
Rout = imageout(:,:,1);
Gout = imageout(:,:,2);
Bout =imageout(:,:,3);
% disp(mean(R - Rout,'all'))
% disp(mean(G - Gout,'all'))
% disp(mean(B - Bout,'all'))

figure;
title('DCT Before and after Reconstruction')
subplot(1,2,1)
imshow(image);

subplot(1,2,2)
imshow(imageout);

mse = mse_image(image, imageout);
InputImage_bytes = size(image,1) * size(image,2) * 3;
total_txed_size_bytes = numbcoeffs * 32;
cr  = InputImage_bytes / total_txed_size_bytes;
disp('Input image size: ' + string(InputImage_bytes/1000) + '[kb]');
disp('transmitted image size: ' + string(total_txed_size_bytes/1000) + '[kb]');
disp("DCT Mean Squared Error: " + num2str(mse));
disp("DCT compression ratio " + num2str(cr));
disp("DCT Structural Similarity Index(ssim): " + num2str(ssim(image,imageout)));
disp("DCT Peak Signal-to-noise ratio (PSNR): " + num2str(psnr(image,imageout)));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% DWT
% Input Image
%im=imread('../../Data/checkerboard_1024x1024','png');
%im=imread('../../Data/dog_1024x1024','png');
%im=imread('../../Data/AbrahamLincoln_1024x1024','png');
%im=imread('../../Data/women_512x512','png');
im = image;
% convert color image to gray scale.
grayScaleImg = rgb2gray(im);

% levels = [1,2,3,4,5,6];
% thresholds = [1 0.5 0.1 0.05 0.01 0.005 0.001 ];
% mse = zeros(length(levels), length(thresholds));
% cr = zeros(length(levels), length(thresholds));
% 
% i = 1;
% for level = levels
%    j = 1;
%    for threshold = thresholds
%        [mse(i,j) cr(i,j)] = waveletProcessing(grayScaleImg, level, threshold);
%        j = j + 1;
%    end
%    i = i +1;
% end
% figure; imagesc((mse)); colorbar; ylabel('level'); xlabel('threshold')
% figure; imagesc((cr)); colorbar; ylabel('level'); xlabel('threshold')
% 

waveletProcessing(grayScaleImg, 5, 0.015 );

%% wavelet transform parameters
function [mse,cr] = waveletProcessing(grayScaleImg, levels, threshold)

% image size calculation
InputImage_bytes = size(grayScaleImg,1) * size(grayScaleImg,2);
disp('Input image size: ' + string(InputImage_bytes/1000) + '[kb]');

%levels = 3;
% haar wavelet

% compute wavelet coefficents DWT
[waveletCoefficents, waveletCoefficentSizes] = customDWT(grayScaleImg, levels);

% Thresholding
waveletCoefficents = thresholdSignal(waveletCoefficents, threshold);

% Entropy encoding
encodedIdxs = (differenceEncode(waveletCoefficents));
nonZeroCoeff = waveletCoefficents(waveletCoefficents ~= 0);

% Quantization 8-bit
[quantizedCoefficents, dr, minInput] = customQuantize(nonZeroCoeff);

% measure size
Idx_size_bytes = length(encodedIdxs) * 2;
coefficent_size_bytes = length(quantizedCoefficents) * 1;
total_txed_size_bytes = Idx_size_bytes + coefficent_size_bytes;
disp('transmitted image size: ' + string(total_txed_size_bytes/1000) + '[kb]');

% Dequantization 8-bit
[dequantizedCoefficents] = inverseCustomQuantize(quantizedCoefficents, dr, minInput);


% entropy decoding
recoveredCoeffcients = reconstructCoefficentMatrix(encodedIdxs, dequantizedCoefficents, length(waveletCoefficents));

% recover image using IDWT
recoveredImage = uint8(customIDWT(recoveredCoeffcients, waveletCoefficentSizes, levels));
figure;
title('DWT Original and Reconstructed')
subplot(1,2,1)
imshow(grayScaleImg)
subplot(1,2,2)
imshow(recoveredImage)

% print MSE
mse = mse_image(grayScaleImg, recoveredImage);
cr  = InputImage_bytes / total_txed_size_bytes;
disp("DWT Mean Squared Error: " + num2str(mse));
disp("DWT compression ratio " + num2str(cr));
disp("DWT Structural Similarity Index(ssim): " + num2str(ssim(grayScaleImg,recoveredImage)));
disp("DWT Peak Signal-to-noise ratio (PSNR): " + num2str(psnr(grayScaleImg,recoveredImage)));

end

%% Helper functions %%
% MSE
function mse = mse_image(imageA, imageB)
    mse = mean((double(imageA) - double(imageB)).^2,"all");
end

%% Thresholding
function output = thresholdSignal(input, threshold)
    sorted_signal = sort(input);
    startIdx = (1-threshold)*length(sorted_signal);
    if(startIdx ~= 0)
        thresholdValue = sorted_signal(floor(startIdx));
        keep_idxs = abs(input) > thresholdValue;
        output = input .* keep_idxs;
    else
        output = input;
    end
end

%% Quantization
function [output, dr, minInput] = customQuantize(input)
    % shift numbers to be positive
    %figure; histogram(input(0~=input))

    minInput = min(input);
    positiveInput = input + -minInput;
    
    % quantize to 8 bits linearly
    dr = max(positiveInput);
    output = uint8(round( 255 * positiveInput / dr ));
    %figure; histogram(output(52~=output))

end

function [output] = inverseCustomQuantize(input, dr, minInput)
    positiveInput = dr * double(input) / 255;
    %figure; histogram(positiveInput(median(positiveInput)~=positiveInput))

    % shift numbers back
    output = positiveInput + minInput;
    %figure; histogram(output(median(output)~=output))

end

%% Entropy

% encoding
function [encodedDifferences] = differenceEncode(data)
    % Find the indices of non-zero elements
    nonZeroIndices = find(data ~= 0);
    
    encodedDifferences = diff(nonZeroIndices);
    encodedDifferences = [nonZeroIndices(1); encodedDifferences];
end

function M = reconstructCoefficentMatrix(indices, data, sz)
    decodedIdxs = differenceDecode(indices);
    
    M = zeros(sz,1);
    M(decodedIdxs) = data;
end

% decoding
function decodedIndices = differenceDecode(encodedDifferences)
    decodedIndices = cumsum(encodedDifferences);
end


%% IDWT Functions 
% input is stored as [A, Hn, Vn, Dn, ... , H1, V1, D1]
function S = customIDWT(M, Book, levels)
lastIdx = 1;
for l=levels:-1:1
    % extract coefficents

    %level size
    rows = Book(l,1);
    cols = Book(l,2);

    coef_size = rows * cols;
    

    if( l == levels) % base case
        A = reshape(M(lastIdx:coef_size+lastIdx - 1), [rows cols]);  lastIdx = lastIdx + coef_size;
    else 
        A = S;
    end
    %figure; imagesc(A);

    H = reshape(M(lastIdx:coef_size+lastIdx - 1), [rows cols]);  lastIdx = lastIdx + coef_size;
    V = reshape(M(lastIdx:coef_size+lastIdx - 1), [rows cols]);  lastIdx = lastIdx + coef_size;
    D = reshape(M(lastIdx:coef_size+lastIdx - 1), [rows cols]);

    lastIdx = lastIdx + coef_size;

    % reconstructed signal is A + Detail_1 + ... + Detail_n
    A_rec = interpolateColumnsAndLowPass(A);
    H_rec = interpolateColumnsAndHighPass(H);

    L_rec = interpolateRowsAndLowPass(A_rec + H_rec);

    D_rec = interpolateColumnsAndHighPass(D);
    V_rec = interpolateColumnsAndLowPass(V);

    H_rec = interpolateRowsAndHighPass(D_rec + V_rec);
    S = H_rec + L_rec;
end




end

function output = interpolateRowsAndLowPass(input)
lowPass = [1 1] / sqrt(2);
output = interpolateRowsAndConv(lowPass, input);
end
function output = interpolateRowsAndHighPass(input)
highPass = [-1 1] / sqrt(2);
output = interpolateRowsAndConv(highPass, input);
end
function output = interpolateColumnsAndLowPass(input)
lowPass = [1 1 ] / sqrt(2);
output = interpolateColumnsAndConv(lowPass, input);
end
function output = interpolateColumnsAndHighPass(input)
highPass = [-1 1] / sqrt(2);
output = interpolateColumnsAndConv(highPass, input);
end

function output=interpolateRowsAndConv(filter,data)
    
    numRows = size(data,1);
    
    for row=1:numRows
        d_sym = data(row,:);
        d_sym = [d_sym(2) d_sym];
        IR = upsample(d_sym, 2);
        IR = conv(IR, filter, "valid");


        IR(1) = [];
        output(row,:) = IR;
    end
end
function output=interpolateColumnsAndConv(filter,data)

    numColumns = size(data,2);

    for column=1:numColumns
        d_sym = data(:,column);
        d_sym = [d_sym(2); d_sym];
        IC = upsample(d_sym,2);
        IC = conv(IC, filter,  "valid");
        IC(1) = [];
        
        output(:,column) = IC;
    end
end


%% DWT Functions
% output is stored as [A, Hn, Vn, Dn, ... , H1, V1, D1]
function [output, D_out] = customDWT(image, level)
output = [];
D_out = [];
level_coef = cell(level);

for l=1:level
    
    L  = lowPassRowsAndDecimate(image);
    LL = lowPassColumnsAndDecimate(L);
    LH = highPassColumnsAndDecimate(L);

    H = highPassRowsAndDecimate(image);
    HH = highPassColumnsAndDecimate(H);
    HL = lowPassColumnsAndDecimate(H);

    A = LL;
    H = LH;
    V = HL;
    D = HH;

    D_out = [D_out; [size(H,1) size(H,2)]];

    A_flat = reshape(A, [size(A,1) * size(A,2) 1]);
    H_flat = reshape(H, [size(H,1) * size(H,2) 1]);
    V_flat = reshape(V, [size(V,1) * size(V,2) 1]);
    D_flat = reshape(D, [size(D,1) * size(D,2) 1]);
    
    if(l == level)
        level_coef(l) = {[A_flat; H_flat; V_flat; D_flat]};
    else
        level_coef(l) = {[H_flat; V_flat; D_flat]};
    end
    image = LL;
    output = [cell2mat(level_coef(l)); output];
end



end

function output = lowPassRowsAndDecimate(input)
lowPass = [1 1] / sqrt(2);
output = convRowsAndDecimate(lowPass, input);
end
function output = lowPassColumnsAndDecimate(input)
lowPass = [1 1] / sqrt(2);
output = convColumnsAndDecimate(lowPass, input);
end
function output = highPassColumnsAndDecimate(input)
highPass = [1 -1]/ sqrt(2);
output = convColumnsAndDecimate(highPass, input);
end
function output = highPassRowsAndDecimate(input)
highPass = [1 -1] / sqrt(2);
output = convRowsAndDecimate(highPass, input);
end

function output=convColumnsAndDecimate(filter,data)
    numColumns = size(data,2);
    
    for column=1:numColumns
        IC = conv(data(:,column), filter,"same");
        IC = downsample(IC,2);
        output(:,column) = IC;
    end

end
function output=convRowsAndDecimate(filter,data)
    numRows = size(data,1);
    
    for row=1:numRows
        IR = conv(data(row,:), filter, "same");
        IR = downsample(IR, 2);
        output(row,:) = IR;
    end

end
%% DCT Functions

function padded_image = pad_image_to_multiple_of_8(image)
    % Get the size of the image
    [rows, cols, channels] = size(image);

    % Calculate the padding required for rows and columns
    pad_rows = mod(32 - mod(rows, 32), 32);
    pad_cols = mod(32 - mod(cols, 32), 32);

    % Pad the image by duplicating edge pixels
    padded_image = padarray(image, [pad_rows, pad_cols], 'replicate', 'post');

    % Display the original and padded image sizes
    fprintf('Original size: %d x %d\n', rows, cols);
    fprintf('Padded size: %d x %d\n', size(padded_image, 1), size(padded_image, 2));
end

function OUT = maxsforwardDCT(A)
[m, n] = size(A);
OUT = zeros(m, n);

for p = 1:m
    for q = 1:n

        if p == 1
            ap = 1 / sqrt(m);
        else
            ap = sqrt(2 / m);
        end
        if q == 1
            aq = 1 / sqrt(n);
        else
            aq = sqrt(2 / n);
        end

        sum = 0;
        for i = 1:m
            for j = 1:n
                sum = sum + A(i, j) * cos(pi * (2 * (i-1) + 1) * (p-1) / (2 * m)) * cos(pi * (2 * (j-1) + 1) * (q-1) / (2 * n));
            end
        end

        OUT(p, q) = ap * aq * sum;
    end
end

end

function OUT = maxsbackwardDCT(A)

[m, n] = size(A);
OUT = zeros(m, n);

for i = 1:m
    for j = 1:n
        sum = 0;
        for p = 1:m
            for q = 1:n
                if p == 1
                    ap = 1 / sqrt(m);
                else
                    ap = sqrt(2 / m);
                end
                if q == 1
                    aq = 1 / sqrt(n);
                else
                    aq = sqrt(2 / n);
                end
                sum = sum + ap * aq * A(p, q) * cos(pi * (2 * (i - 1) + 1) * (p - 1) / (2 * m)) * cos(pi * (2 * (j - 1) + 1) * (q - 1) / (2 * n));
            end
        end
        OUT(i, j) = sum;
    end
end

end

function Q = jpeg_qtable(quality)
    % Standard JPEG quantization table
    Q50 = [16 11 10 16 24 40 51 61;
           12 12 14 19 26 58 60 55;
           14 13 16 24 40 57 69 56;
           14 17 22 29 51 87 80 62;
           18 22 37 56 68 109 103 77;
           24 35 55 64 81 104 113 92;
           49 64 78 87 103 121 120 101;
           72 92 95 98 112 100 103 99];
    
    if quality < 50
        scale = 5000 / quality;
    else
        scale = 200 - 2 * quality;
    end
    
    Q = floor((Q50 * scale + 50) / 100);
    Q(Q < 1) = 1;
    Q(Q > 255) = 255;
end
