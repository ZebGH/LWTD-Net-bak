path = 'C:\Users\Zeb\Desktop\pic\VID_20220118_092525\';
 fileExt = '*.png';
 files = dir(fullfile(path,fileExt));
 len = size(files,1);
 b = 0;
 for i=1:len
 fileName = strcat(path,files(i,1).name),
%1. Load the image, for example
    image        = imread(fileName);
%1. Load the image, for example
  %image         = imread('109_derain.png'); 
%2. Call this function to calculate the quality score:
   qualityscore = brisquescore(image)
    b = b+ qualityscore
 end
 c = b / len