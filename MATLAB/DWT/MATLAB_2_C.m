clear
close all
% im=imread('C:/Users/lstup.LANCE/source/repos/ECE529/ECE529-Project/Data/Checkerboard_pattern.png');
% grayScaleImg = double(im);

%% Input Image
im=imread('C:/Users/lstup.LANCE/source/repos/ECE529/ECE529-Project/Data/paris-1213603.jpg');
grayScaleImg = rgb2gray(im);

InputImage_bytes = size(grayScaleImg,1) * size(grayScaleImg,2);
disp('Input image size: ' + string(InputImage_bytes/1000) + '[kb]');

figure; image(grayScaleImg);

%% wavelet transform 2D haar
% lets use symetic padding
img = padarray(grayScaleImg, [1 1],"symmetric");

[M] = customDWT(img,1);
%[~, H2, V2, D2] = customDWT(A1);
%figure; imagesc(A1);


%dec1 = [A1 H1; V1 D1];
%figure; image(dec1);

S = uint8(customIDWT(A1,H1,V1,D1));

mse = mse_image(grayScaleImg, S);


figure; image(S);
disp("Mean Squared Error: " + num2str(mse));



%% Helper functions %%
% MSE
function mse = mse_image(imageA, imageB)
    mse = mean((double(imageA) - double(imageB)).^2,"all");
end

% IDWT Functions 
% input is stored as [A, Hn, Vn, Dn, ... , H1, V1, D1]
function S = customIDWT(A, H, V, D)

% reconstructed signal is A + Detail_1 + ... + Detail_n
A_rec = interpolateColumnsAndLowPass(A);
H_rec = interpolateColumnsAndHighPass(H);

L_rec = interpolateRowsAndLowPass(A_rec + H_rec);

D_rec = interpolateColumnsAndHighPass(D);
V_rec = interpolateColumnsAndLowPass(V);

H_rec = interpolateRowsAndHighPass(D_rec + V_rec);

S = H_rec + L_rec;

S(size(S,1),:) = [];
S(:,size(S,2)) = [];


end

function output = interpolateRowsAndLowPass(input)
lowPass = [1 1] / sqrt(2);
output = interpolateAndConv(lowPass, input, 'rows');
end
function output = interpolateRowsAndHighPass(input)
highPass = [-1 1] / sqrt(2);
output = interpolateAndConv(highPass, input, 'rows');
end
function output = interpolateColumnsAndLowPass(input)
lowPass = [1 1 ] / sqrt(2);
output = interpolateAndConv(lowPass, input, 'columns');
end
function output = interpolateColumnsAndHighPass(input)
highPass = [-1 1] / sqrt(2);
output = interpolateAndConv(highPass, input, 'columns');
end

function output=interpolateAndConv(filter,data,Dim)

numRows = size(data,1);
numColumns = size(data,2);

if(Dim == "rows")

    for row=1:numRows
        IR = upsample(data(row,:), 2);
        IR = conv(IR, filter, "valid");
        output(row,:) = IR;
    end
else
    for column=1:numColumns
        IC = upsample(data(:,column),2);
        IC = conv(IC, filter, "valid");
        output(:,column) = IC;
    end
end

end

% DWT Functions
% output is stored as [A, Hn, Vn, Dn, ... , H1, V1, D1]
function [output] = customDWT(image, level)
output = [];
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

    A_flat = reshape(A, [size(A,1) * size(A,2) 1]);
    H_flat = reshape(H, [size(H,1) * size(H,2) 1]);
    V_flat = reshape(V, [size(V,1) * size(V,2) 1]);
    D_flat = reshape(D, [size(D,1) * size(D,2) 1]);

    if(l == level)
        level_coef(l) = {[A_flat; H_flat; V_flat; D_flat]};
    else
        level_coef(l) = [H_flat V_flat D_flat];
    end
    
    output = [cell2mat(level_coef(l)) output];
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
        IR = conv(data(row,:), filter,"valid");
        %IR = circshift(IR,1);
        IR = downsample(IR, 2);
        output(row,:) = IR;
    end
else
    for column=1:numColumns
        IC = conv(data(:,column), filter,"valid");
        %IC = circshift(IC,1);
        IC = downsample(IC,2);
        output(:,column) = IC;
    end
end

end
