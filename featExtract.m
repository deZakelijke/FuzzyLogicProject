function features = featExtract(image)
    lines = houghExtract(image);
   
    features = lines;
end