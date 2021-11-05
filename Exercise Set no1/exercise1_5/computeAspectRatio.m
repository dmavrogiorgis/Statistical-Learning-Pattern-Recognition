function aRatio = computeAspectRatio(image)
    %[num_rows, num_cols] = size(image);

    % Fill your code
    sum_row = sum(image,2);
    sum_col = sum(image,1);
    
    min_row = find(sum_row,1);
    max_row = find(sum_row,1,'last');
    min_col = find(sum_col,1);
    max_col = find(sum_col,1,'last');
    
    width = max_col-min_col;
    height = max_row-min_row;
    
    aRatio = width/height;
   
end