%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% ����ͼ���ƽ���ݶ�%% %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ���ߣ������� ʱ�䣺2008.6.9 %%
%%%%%%%% QQ��76424677   %%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%     ����ı�δ����       %%%

%��������ͼ��
path = 'G:\lane\NIQE\test_imgs\565\';
fileExt = '*.jpg';
files = dir(fullfile(path,fileExt));
len = size(files,1);
xx = 0
for i=1:len
fileName = strcat(path,files(i,1).name),


%img=imread('101_derain.png')
img=imread(fileName)

%����ת��
img = double(img);

%��ȡͼ���С��Ϣ
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
%��ͼ���ƽ���ݶ�
%MeanGradient1= mean(g);
xxx = xx / len
