% Made by Micha de Groot, Amir al Normani and Bjarne de Jong

imageTrainFile = 'train-images-idx3-ubyte';
labelTrainFile = 'train-labels-idx1-ubyte';
%[images, labels] = readMNIST(imageTrainFile, labelTrainFile, 10, 0);
images = loadMNISTImages(imageTrainFile);
labels = loadMNISTLabels(labelTrainFile);
display_network(images(:,1:20));
%disp(labels(2:2));

firstImage = reshape(images(:,2),28,28);

circles = circleExtract(firstImage);
%features = featExtract(firstImage);