%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% 计算图像的平均梯度%% %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 作者：刘建华 时间：2008.6.9 %%
%%%%%%%% QQ：76424677   %%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%     共享改变未来！       %%%

%读入两幅图像
path = 'G:\lane\NIQE\test_imgs\565\';
fileExt = '*.jpg';
files = dir(fullfile(path,fileExt));
len = size(files,1);
xx = 0
for i=1:len
fileName = strcat(path,files(i,1).name),


%img=imread('101_derain.png')
img=imread(fileName)

%精度转换
img = double(img);

%获取图像大小信息
    [r,c,b] = size(img);
    dx = 1;
    dy = 1;
    
    for k = 1 : b
        band = img(:,:,k);
        dzdx=0.0;
        dzdy=0.0;
        [dzdx,dzdy] = gradient(band,dx,dy);
        s = sqrt((dzdx .^ 2 + dzdy .^2) ./ 2);
        g(k) = sum(sum(s)) / ((r - 1) * (c - 1));
        %MeanGradient = mean(g)
        
    end
    MeanGradient= mean(g)
    xx = MeanGradient + xx
end
%求图像的平均梯度
%MeanGradient1= mean(g);
xxx = xx / len
