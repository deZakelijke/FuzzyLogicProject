% Made by Micha de Groot, Amir al Normani and Bjarne de Jong

imageTrainFile = 'train-images-idx3-ubyte';
labelTrainFile = 'train-labels-idx1-ubyte';
%[images, labels] = readMNIST(imageTrainFile, labelTrainFile, 10, 0);
images = loadMNISTImages(imageTrainFile);
labels = loadMNISTLabels(labelTrainFile);
%display_network(images(:,1:30));
%disp(labels(2:2));

testImageID = 5;

firstImage = imresize(reshape(images(:,testImageID),28,28),2);

%BW = edge(firstImage, 'Canny', [0.8 0.9]);
Bw = imsharpen(firstImage);
%BW = firstImage;
imshow(BW);
hold on;
%circles = circleExtract(BW);
%features = featExtract(BW);
