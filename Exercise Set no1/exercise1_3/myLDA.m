function A = myLDA(Samples, Labels, NewDim)
% Input:    
%   Samples: The Data Samples 
%   Labels: The labels that correspond to the Samples
%   NewDim: The New Dimension of the Feature Vector after applying LDA
    
	[NumSamples,NumFeatures] = size(Samples);
    
	A = zeros(NumFeatures,NewDim);
    NumLabels = length(Labels);
    if(NumSamples ~= NumLabels)
        fprintf('\nNumber of Samples are not the same with the Number of Labels.\n\n');
        exit
    end
    Classes = unique(Labels);
    NumClasses = length(Classes);  %The number of classes

    %For each class i
	%Find the necessary statistics

    P = zeros(1, NumClasses);
    mu = zeros(NumFeatures, NumClasses);
    Sw = zeros(NumFeatures, NumFeatures);
    for i = 1:NumClasses 
        %Calculate the Class Prior Probability
        P(i) = sum(Labels==(i-1))/NumLabels;
        
        %Calculate the Class Mean 
        mu(:,i) = mean(Samples(Labels==(i-1),:));
        
        %Calculate the Within Class Scatter Matrix
        Sw = Sw + P(i)*cov(Samples(Labels==(i-1),:));
    end
    
    %Calculate the Global Mean
    m0 = mean(mu);
 
    %Calculate the Between Class Scatter Matrix
	Sb = zeros(NumFeatures, NumFeatures);
    for i=1:NumClasses %For all classes
        Sb = Sb+P(i)*(mu(:,i)-m0)*(mu(:,i)-m0)';
    end

    %Eigen matrix EigMat=inv(Sw)*Sb
    EigMat = inv(Sw)*Sb;
    
    %Perform Eigendecomposition
    [V,D] = eig(EigMat);
    
    eigenval = diag(D);                       %Vector of eigenvalues
    [~, ind] = sort(eigenval,1,'descend');    %Sort them
    eigenvec = V(:, ind);                     %Corresponding eigenvectors
    
    %Select the NewDim eigenvectors corresponding to the top NewDim
    %eigenvalues (Assuming they are NewDim<=NumClasses-1)
	%% You need to return the following variable correctly.
	A = zeros(NumFeatures, NewDim);  % Return the LDA projection vectors
    
    A = eigenvec(:,1:NewDim); 
