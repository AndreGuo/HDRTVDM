function [degree, percentage, varargout] = HDRdegreeAssessment(img, varargin)
    % Copyright: guocheng@cuc.edu.cn
    % First ver. 28 Mar 2022
    % Last modified: 14 Jun 2023
    %
    % Input argsuments:
    %  Required (1):
    % 'img'           - m-by-n-by-3 RGB image array: with
    %                   BT.2020 primaries, (non)linear, normalized to [0,1]
    %                   SHOULD BE: single | double
    %  Optional (5):
    % 'non_linearity' - char: the EOTF of 'img'
    %                   'PQ' (default) | 'HLG' | 'gamma' | 'linear'
    %                   ('gamma' non-linearity actually do not exist in HDR
    %                   standard, here we assume the peak luminance of
    %                   gamma-encoded HDR image is 1000nit)
    % 'defuse_white'  - num:
    %                   the assummed luminance of defuse white in HDR image
    %                   100 (default) | 203 | 300
    % 'peak_luminance'- num:
    %                   1000 (default) | 2000 | 4000 | 10000
    %                   see LINE XXX for detailed explian
    % 'output_truncated': (our extension)
    %                   enter the filename TO output the image with
    %                   highlight value truncated.
    % 'output_heatmap': (our extension)
    %                   enter the filename TO output a normalized heatmap
    %                   telling the position and degree of highlight pixels
    %
    % Onput argsuments:
    % [degree, percentage]
    %     when 'output_truncated' & 'output_heatmap' == false
    % [degree, percentage, TruncatedHDR]
    %     when 'output_truncated' == true & 'output_heatmap' == false
    % [degree, percentage, ~, Heatmap]
    %     when 'output_truncated' == false & 'output_heatmap' == true
    % [degree, percentage, TruncatedHDR, Heatmap]
    %     when 'output_truncated' == true & 'output_heatmap' == true
    %
    % Useage:
    % [EHL, FHLP] = HDRdegreeAssessment(img);
    %  to calculate EHL and FHLP only

    p = inputParser;
    addRequired(p,'img',@(x)validateattributes(x,...
        {'numeric'},{'size',[NaN,NaN,3]}))
    addOptional(p,'non_linearity','PQ',@(x)validateattributes(x,...
        {'char'},{'nonempty'}))
    addOptional(p,'defuse_white',100,@(x)validateattributes(x,...
        {'numeric'},{'nonempty'}))
    addOptional(p,'peak_luminance',1000,@(x)validateattributes(x,...
        {'numeric'},{'nonempty'}))
    addOptional(p,'output_truncated',false,@(x)validateattributes(x,...
        {'logical'},{'nonempty'}))
    addOptional(p,'output_heatmap',false,@(x)validateattributes(x,...
        {'logical'},{'nonempty'}))
    parse(p,img,varargin{:})

    switch p.Results.non_linearity
        case 'PQ'
            m1 = 2610/16384; m2 = 2523/32;...
                c1 = 3424/4096; c2 = 2413/128; c3 = 2392/128;
            eotf = @(x)((max((x.^(1/m2)-c1), zeros(size(x)))...
                ./(c2-c3.*(x.^(1/m2)))).^(1/m1));
        case 'HLG'
            a = 0.17883277; b = 0.28466892; c = 0.55991073;
            eotf = @(x)(((x.^2)/3).*(x>=0 & x<=1/2) +...
                ((exp((x-c)/a)+b)/12).*(x>1/2 & x<=1));
        case 'gamma'
            eotf = @(x)(x.^(1/0.45));
        case 'linear'
            eotf = @(x)(x);
        otherwise
            error('Unsupported non-linearity!')
    end

    rgb_ = img;
    % To linear-light
    rgb = eotf(rgb_);

    % We treat values bigger than 'p.Results.peak_luminance'
    % as highligh part of an HDR frame.
    DefuseWhiteNormVal = p.Results.defuse_white/p.Results.peak_luminance;
    % NOTE THAT:
    % 1. HLG and gamma are realtive luminance where normalized 1 represent
    %    max diaplay lumiance (e.g. 1000nit)
    % 2. PQ system uses absolute lumiannce where normalized 1 always means
    %    10000nit, but we also allow cases e.g. the non-linearity is PQ but
    %    lumiance is relative (though it is not the commom case), so we let
    %    'peak_luminance' up to your choise.
    % 3. Overall, 'peak_luminance' represent what normalized 1 means in our
    %    own system.

    
    % Truncate all highligh value to defuse white
    rgbTruncated = rgb;
    rgbTruncated(rgbTruncated>DefuseWhiteNormVal) = DefuseWhiteNormVal;

    numHighlight = length(rgbTruncated(rgbTruncated==DefuseWhiteNormVal));
    percentage = numHighlight/numel(rgb);

    distance = sqrt((rgb-rgbTruncated).^2);
    % We treat e.g. (0.203,0.203,0.203) vs (1,1,1) as max distance
    maxDistance = sqrt((1-DefuseWhiteNormVal)^2);
    degree = mean(distance(:))/maxDistance;


    if p.Results.output_truncated == true
        switch p.Results.non_linearity
            case 'PQ'
                oetf = @(x)(((c1+c2*(x.^m1))./(1+c3*(x.^m1))).^m2);
            case 'HLG'
                oetf = @(x)((sqrt(3*x)).*(x>=0 & x<=1/12) +...
                    (a*log(12*x-b)+c).*(x>1/12 & x<=1));
            case 'gamma'
                oetf = @(x)(x.^0.45);
            case 'linear'
                oetf = @(x)(x);
                warning('You are returning a linear image.');
            otherwise
                error('Unsupported OETF!');
        end
        rgbTruncated_ = oetf(rgbTruncated);
        varargout{1} = rgbTruncated_;
    end

    if p.Results.output_heatmap == true
        varargout{2} = distance/maxDistance; % which can be later used:
        % imwrite(varargout{2}, trubo(60), 'name.jpg') in breakpoint
    end
