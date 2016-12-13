function circle = circleExtract(image)
    [centers, radii] = imfindcircles(image, [12,20], 'Sensitivity', 0.92);
    
    imshow(image)
    h = viscircles(centers, radii);
    circle = 0;
end