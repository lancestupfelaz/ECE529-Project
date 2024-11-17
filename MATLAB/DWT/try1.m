%% Loading of an image
im=imread('C:/Users/lstup.LANCE/source/repos/ECE529/ECE529-Project/Data/paris-1213603.jpg');
% isolate channels
r = im(:,:,1);
g = im(:,:,2);
b = im(:,:,3);

[c,s] = wavedec2(r,2,'haar');
A1 = appcoef2(c,s,'haar',1);
A1img = wcodemat(A1,255,'mat',1);
A1img = uint8(A1img);

flatA1img = reshape(A1img, [1 511*338]);
flatA1img = wcodemat(flatA1img,255,'mat',1);
h = histogram(flatA1img,"BinWidth",1.0001);
counts = double(h.Values);

%% Hoffman code
dict = huffmandict(1:h.NumBins,counts / length(flatA1img));
code_hoff = huffmanenco(flatA1img,dict);
dseq = huffmandeco(code,dict);
dseq = uint8(dseq);


%% Arithmatic code
counts = nonzeros(counts);
code_arith = arithenco(flatA1img, counts);
%dseq = arithdeco(code,counts,length(flatA1img));
%dseq = uint8(dseq);

%% plotting and efficency




%% image rec

out = waverec2(c,s,'haar');
figure;
imagesc(ouput)
colormap pink(255)
title('Approximation Coef. of Level 1')

%% stats

disp("Hoffman encoding compression: " + string((length(code_hoff)/8) / (511*338)) + "%")
disp("Arithemtic encoding compression: " + string((length(code_arith)/8) / (511*338)) + "%")

