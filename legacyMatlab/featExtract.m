function features = featExtract(image)
    % The function that calls all the feature extract functions and checks
    % for decent confidence values
    
    lines = houghExtract(image);
    [fitobject1, gof1, fitobject2, gof2] = curveExtract(image);
    
    features = lines;
end