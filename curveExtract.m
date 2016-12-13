function [fitobject1, gof1, fitobject2, gof2] = curveExtract(image)
    % Fits two curves to the image and returns the curves and a confidence
    % score.

    X1 = [];
    Y1 = [];
    X2 = [];
    Y2 = [];
    for i=1:length(image)
        index = find(image(:,i));
        if index
            X1 = [X1 i];
            Y1 = [Y1 index(1)];
            X2 = [X2 i];
            Y2 = [Y2 index(end)];
        end
    end
    [fitobject1, gof1] = fit(X1', Y1', 'poly2');
    [fitobject2, gof2] = fit(X2', Y2', 'poly2');
    hold on
    plot(fitobject1);
    legend('off');
    hold on
    plot(fitobject2);
    legend('off');

end
