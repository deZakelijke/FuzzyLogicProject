% Made by Micha de Groot, Amir al Normani and Bjarne de Jong

imageTrainFile = 'train-images-idx3-ubyte';
labelTrainFile = 'train-labels-idx1-ubyte';
%[images, labels] = readMNIST(imageTrainFile, labelTrainFile, 10, 0);
images = loadMNISTImages(imageTrainFile);
labels = loadMNISTLabels(labelTrainFile);
display_network(images(:,1:30));
%disp(labels(2:2));

firstImage = imresize(reshape(images(:,3),28,28),2);

BW = edge(firstImage);
%imshow(BW);
%circles = circleExtract(BW);
features = featExtract(BW);