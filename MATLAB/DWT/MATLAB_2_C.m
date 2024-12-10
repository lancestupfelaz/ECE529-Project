clear
close all

%% Input Image
%im=imread('../../Data/checkerboard_1024x1024','png');
%im=imread('../../Data/dog_1024x1024','png');
%im=imread('../../Data/AbrahamLincoln_1024x1024','png');
%im=imread('../../Data/women_512x512','png');

% convert color image to gray scale.
%grayScaleImg = rgb2gray(im);
grayScaleImg = [1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16 ];
grayScaleImg = reshape(1:64, 8, 8)';

%figure; imshow(grayScaleImg)

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

waveletProcessing(grayScaleImg, 2, 1 );

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
figure; imshow(recoveredImage)

% print MSE
mse = mse_image(grayScaleImg, recoveredImage);
cr  = InputImage_bytes / total_txed_size_bytes;
disp("Mean Squared Error: " + num2str(mse));
disp("compression ratio " + num2str(cr));
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
