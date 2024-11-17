clear
%% Loading of an image
im=imread('C:/Users/lstup.LANCE/source/repos/ECE529/ECE529-Project/Data/paris-1213603.jpg');

r = rgb2gray(im);

InputImage_bytes = size(r,1) * size(r,2);
disp('Input image size: ' + string(InputImage_bytes/1000) + '[kb]');
figure;

title('Input Image')
%r = reshape(r, [size(r,1)*size(r,2)]);
[C,S] = wavedec2(r,4,'db1');
Csort = sort(abs(C(:)));

counter = 1;
for keep = [0.1 0.05 0.01 0.005]
    subplot(2,2,counter);
    thresh = Csort(floor((1-keep)*length(Csort)));
    ind = abs(C) > thresh;
    Cfilt = C.*ind;
    sparse_C = sparse(Cfilt);

    % plot
    Arecon = uint8(waverec2(Cfilt,S, 'db1'));
    imshow(256-uint8(Arecon));
    title(['', num2str(keep*100), '%']);
    counter = counter + 1;
end

compressedImg = [uint32(idxs); single(vals)];
TransmittedImage_bytes = length(vals) * 4 * 2;
disp('Transmitted image size: ' + string(TransmittedImage_bytes/1000) + '[kb]');
