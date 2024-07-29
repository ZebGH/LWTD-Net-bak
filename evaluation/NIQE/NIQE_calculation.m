
 load modelparameters.mat
 
 blocksizerow    = 96;
 blocksizecol    = 96;
 blockrowoverlap = 0;
 blockcoloverlap = 0;

 path = 'C:\Users\Zeb\Desktop\pic\';
 fileExt = '*.jpg';
 files = dir(fullfile(path,fileExt));
 len = size(files,1);
 b = 0
 for i=1:len
 fileName = strcat(path,files(i,1).name),
 
 
im =imread(fileName);

quality = computequality(im,blocksizerow,blocksizecol,blockrowoverlap,blockcoloverlap, ... 
    mu_prisparam,cov_prisparam)
b = b + quality
 end
 c = b /len

% im =imread('101_derain.png');
% 
% quality2 = computequality(im,blocksizerow,blocksizecol,blockrowoverlap,blockcoloverlap, ...
%     mu_prisparam,cov_prisparam)
% 
% im =imread('102.png');
% 
% quality3 = computequality(im,blocksizerow,blocksizecol,blockrowoverlap,blockcoloverlap, ...
%     mu_prisparam,cov_prisparam)
% 
% 
% im =imread('102_derain.png');
% 
% quality4 = computequality(im,blocksizerow,blocksizecol,blockrowoverlap,blockcoloverlap, ...
%     mu_prisparam,cov_prisparam)
% 
% im =imread('103.png');
% 
% quality5 = computequality(im,blocksizerow,blocksizecol,blockrowoverlap,blockcoloverlap, ...
%     mu_prisparam,cov_prisparam)
% 
% im =imread('103_derain.png');
% 
% quality6 = computequality(im,blocksizerow,blocksizecol,blockrowoverlap,blockcoloverlap, ...
%     mu_prisparam,cov_prisparam)
% 
% im =imread('104.png');
% 
% quality7 = computequality(im,blocksizerow,blocksizecol,blockrowoverlap,blockcoloverlap, ...
%     mu_prisparam,cov_prisparam)
% 
% im =imread('104_derain.png');
% 
% quality8 = computequality(im,blocksizerow,blocksizecol,blockrowoverlap,blockcoloverlap, ...
%     mu_prisparam,cov_prisparam)
% 
% im =imread('105.png');
% 
% quality9 = computequality(im,blocksizerow,blocksizecol,blockrowoverlap,blockcoloverlap, ...
%     mu_prisparam,cov_prisparam)
% 
% im =imread('105_derain.png');
% 
% quality10 = computequality(im,blocksizerow,blocksizecol,blockrowoverlap,blockcoloverlap, ...
%     mu_prisparam,cov_prisparam)
% 
% im =imread('106.png');
% 
% quality11 = computequality(im,blocksizerow,blocksizecol,blockrowoverlap,blockcoloverlap, ...
%     mu_prisparam,cov_prisparam)
% 
% im =imread('106_derain.png');
% 
% quality12 = computequality(im,blocksizerow,blocksizecol,blockrowoverlap,blockcoloverlap, ...
%     mu_prisparam,cov_prisparam)
