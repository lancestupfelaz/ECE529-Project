clear
close all
% im=imread('C:/Users/lstup.LANCE/source/repos/ECE529/ECE529-Project/Data/Checkerboard_pattern.png');
% grayScaleImg = double(im);

%% Input Image
%im=imread('../../Data/checkerboard_1024x1024','png');
im=imread('../../Data/dog_1024x1024','png');
grayScaleImg = rgb2gray(im);

InputImage_bytes = size(grayScaleImg,1) * size(grayScaleImg,2);
disp('Input image size: ' + string(InputImage_bytes/1000) + '[kb]');

figure; image(grayScaleImg);

%% wavelet transform 2D haar
% lets use symetic padding
img = padarray(grayScaleImg, [1 1],"symmetric");
%img = grayScaleImg;
levels = 2;
[M, D] = customDWT(grayScaleImg, levels);
%[~, H2, V2, D2] = customDWT(A1);
%figure; imagesc(A1);


%dec1 = [A1 H1; V1 D1];
%figure; image(dec1);

S = uint8(customIDWT(M, D, levels));
figure; image(S);
figure; imagesc(grayScaleImg - S)
mse = mse_image(grayScaleImg, S);


disp("Mean Squared Error: " + num2str(mse));



%% Helper functions %%
% MSE
function mse = mse_image(imageA, imageB)
    mse = mean((double(imageA) - double(imageB)).^2,"all");
end

%         256         169   approx
%         256         169   level 2
%         511         338   level 1
%        1022         676   original image

%    256*169*4+511*338*3 = 691210

%    691210
% IDWT Functions 
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
        %A(:,size(A,2)) = [];
        Rv = 'same';
        Cv = 'same';
    else 
        A = S;
        Rv = 'same';
        Cv = 'same';
    end
    %figure; imagesc(A);

    H = reshape(M(lastIdx:coef_size+lastIdx - 1), [rows cols]);  lastIdx = lastIdx + coef_size;
    V = reshape(M(lastIdx:coef_size+lastIdx - 1), [rows cols]);  lastIdx = lastIdx + coef_size;
    D = reshape(M(lastIdx:coef_size+lastIdx - 1), [rows cols]);

    lastIdx = lastIdx + coef_size;

    % reconstructed signal is A + Detail_1 + ... + Detail_n
    A_rec = interpolateColumnsAndLowPass(A,'same');
    %figure; imagesc(A_rec);
    H_rec = interpolateColumnsAndHighPass(H,'same');

    L_rec = interpolateRowsAndLowPass(A_rec + H_rec,'same');
    %figure; imagesc(L_rec);

    D_rec = interpolateColumnsAndHighPass(D,'same');
    V_rec = interpolateColumnsAndLowPass(V,'same');

    H_rec = interpolateRowsAndHighPass(D_rec + V_rec,'same');
    %figure; imagesc(H_rec);

    S = H_rec + L_rec;
end




end

function output = interpolateRowsAndLowPass(input, valid)
lowPass = [1 1] / sqrt(2);
output = interpolateAndConv(lowPass, input, 'rows', valid);
end
function output = interpolateRowsAndHighPass(input, valid)
highPass = [-1 1] / sqrt(2);
output = interpolateAndConv(highPass, input, 'rows', valid);
end
function output = interpolateColumnsAndLowPass(input, valid)
lowPass = [1 1 ] / sqrt(2);
output = interpolateAndConv(lowPass, input, 'columns', valid);
end
function output = interpolateColumnsAndHighPass(input, valid)
highPass = [-1 1] / sqrt(2);
output = interpolateAndConv(highPass, input, 'columns',valid);
end

function output=interpolateAndConv(filter,data,Dim,valid)

numRows = size(data,1);
numColumns = size(data,2);

if(Dim == "rows")

    for row=1:numRows
        d_sym = data(row,:);
        %d_sym(length(d_sym) + 1) = d_sym(length(d_sym)-1);
        d_sym = [d_sym(2) d_sym];


        IR = upsample(d_sym, 2);
        IR = conv(IR, filter, "valid");
        %IR(length(IR)) = [];
        %IR(length(IR) - 1) = [];
        %IR = circshift(IR,1);

        IR(1) = [];
        output(row,:) = IR;
    end
else
    for column=1:numColumns
        d_sym = data(:,column);
        %d_sym(length(d_sym) + 1) = d_sym(length(d_sym)-1);
        d_sym = [d_sym(2); d_sym];


        IC = upsample(d_sym,2);
        IC = conv(IC, filter,  "valid");
        %IC(length(IC)) = [];
        %IC(length(IC) - 1) = [];
        IC(1) = [];
        %IC = circshift(IC,1);
        
        output(:,column) = IC;
    end
end

end

%         256         169   approx
%         256         169   level 2
%         511         338   level 1
%        1022         676   original image

%    256*169*4+511*338*3 = 691210

% DWT Functions
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
    
    %figure; imagesc(reshape(A_flat, [size(A,1) size(A,2)]));

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
output = convAndDecimate(lowPass, input, 'rows');
end
function output = lowPassColumnsAndDecimate(input)
lowPass = [1 1] / sqrt(2);
output = convAndDecimate(lowPass, input, 'columns');
end
function output = highPassColumnsAndDecimate(input)
highPass = [1 -1]/ sqrt(2);

output = convAndDecimate(highPass, input,'columns');
end
function output = highPassRowsAndDecimate(input)
highPass = [1 -1] / sqrt(2);
output = convAndDecimate(highPass, input,'rows');
end

function output=convAndDecimate(filter,data,Dim)

numRows = size(data,1);
numColumns = size(data,2);

if(Dim == "rows")

    for row=1:numRows
        IR = conv(data(row,:), filter, "same");
        IR = downsample(IR, 2);
        output(row,:) = IR;
    end
else
    for column=1:numColumns
        IC = conv(data(:,column), filter,"same");
        IC = downsample(IC,2);
        output(:,column) = IC;
    end
end

end
