function lines = houghExtract(image)
    BW = edge(image, 'canny');
    [H,T,R] = hough(BW);
    %imshow(H,[],'XData', T, 'YData', R, 'InitialMagnification', 'fit');
    %xlabel('\theta'), ylabel('\rho');
    %axis on, axis normal, hold on;
   
    P = houghpeaks(H,3, 'threshold', ceil(0.3*max(H(:))), 'NHoodSize', [31,31]);
    %x = T(P(:,2)); 
    %y = R(P(:,1));
    %plot(x,y,'s', 'color','white');
    %hold off;
    
    hold off;
    
    lines = houghlines(BW, T, R, P, 'FillGap', 100, 'MinLength',7);
    figure, imshow(image), hold on;
    for k = 1:length(lines)
        xy = [lines(k).point1; lines(k).point2];
        plot(xy(:,1), xy(:,2), 'LineWidth', 2, 'Color', 'green');
    end
    
end