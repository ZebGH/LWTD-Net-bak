 dirpath = "D:\PythonProjects\Zeb'sProjects\PyTorch-Image-Dehazing-master\dehazed_images_0524";
 path0 = dir(dirpath)
 path = 
 fileExt = '*.jpg';
 files = dir(fullfile(path,fileExt));
 len = size(files,1);
 b = 0;
 for i=1:len
 fileName = strcat(path,files(i,1).name);
%1. Load the image, for example
    image        = imread(fileName);
%2. Call this function to calculate the quality score:
    qualityscore = SSEQ(image);
    b = b+ qualityscore;
    
 end
 c = b / len