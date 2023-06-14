function [degree, percentage, varargout] = WCGdegreeAssessment(img, varargin)
    % Copyright: guocheng@cuc.edu.cn
    % First ver. 1 Dec 2021
    % Last modified: 14 Jun 2023
    %
    % Un-official implementation of METHOD 2 "gamut degree assessment" in
    % Bai et al.'s paper "Analysis of high dynamic range and wide color
    % gamut of UHDTV", with some functional extension.
    % DOI: 10.1109/IAEAC50856.2021.9390848
    %
    % Input argsuments:
    %  Required (1):
    % 'img'           - m-by-n-by-3 RGB image array: with
    %                   BT.2020 primaries, nonlinear, normalized to [0,1]
    %                   SHOULD BE: single | double
    %  Optional (6):
    % 'limit_range'   - bool:
    %                   false (default) | true (for some TV exhancge image)
    % 'non_linearity' - char: the EOTF of 'img'
    %                   'PQ' (default) | 'HLG' | 'gamma'
    % 'target_gamut'  - char: the narrow gamut to hard-clip to.
    %                   bt709 (default, as oringinal paper) |
    %                   srgb (same as above) | adobergb (our extension)
    % 'compare_mode'  - char:
    %                   on which color space the distance will caluculate
    %                   'XYZ' (default, as oringinal paper) |
    %                   'xy' (our extension) | 'Yxy' (same as 'xy')
    % 'output_clipped709': (our extension)
    %                    enter the filename TO output the image with
    %                   'target_gamut' hard-clipped from BT.2020 gamut.
    %                   .png format is recommended.
    % 'output_oog_heatmap': (our extension)
    %                    enter the filename TO output a normalized heatmap
    %                    telling the position and degree of OOG (out of
    %                    gamut) or so-called hard-clipped pixels.
    %
    % Onput argsuments:
    % (degree and percentage are respectively EWG and FWGP)
    % [degree, percentage]
    %   when 'output_clipped709' & 'output_oog_heatmap' == false
    % [degree, percentage, HardChipped709]
    %   when 'output_clipped709' == true & 'output_oog_heatmap' == false
    % [degree, percentage, ~, OOGHeatmap]
    %   when 'output_clipped709' == false & 'output_oog_heatmap' == true
    % [degree, percentage, HardChipped709, OOGHeatmap]
    %   when 'output_clipped709' == true & 'output_oog_heatmap' == true
    %
    % Useage:
    % [EWG, FWGP] = WCGdegreeAssessment(img);
    %  to calculate EWG and FWGP only
    %
    % Note:
    %  1. This function requires a MATLAB version >= R2020b;
    % (If you wish a version that works < R2020b, you can issue us at
    % GitHub and we will release one.)
    %  2. It cooperates well with plotImgChromaticity() in breakpoint.

    p = inputParser;
    addRequired(p,'img',@(x)validateattributes(x,...
        {'numeric'},{'size',[NaN,NaN,3]}))
    addOptional(p,'limit_range',false,@(x)validateattributes(x,...
        {'logical'},{'nonempty'}))
    addOptional(p,'non_linearity','PQ',@(x)validateattributes(x,...
        {'char'},{'nonempty'}))
    addOptional(p,'target_gamut','bt709',@(x)validateattributes(x,...
        {'char'},{'nonempty'}))
    addOptional(p,'compare_mode','XYZ',@(x)validateattributes(x,...
        {'char'},{'nonempty'}))
    addOptional(p,'output_clipped709',false,@(x)validateattributes(x,...
        {'logical'},{'nonempty'}))
    addOptional(p,'output_oog_heatmap',false,@(x)validateattributes(x,...
        {'logical'},{'nonempty'}))
    parse(p,img,varargin{:})

    % PATH 1 (above path in paper's Fig. 9): XYZ after hard-chip
    rgb2020 = double(img)*(2^12-1); % [0,1] to [0,4095]
    if p.Results.limit_range == false
        full_2_limited = @(x)(0.85546875*x+256);
        rgb2020 = uint16(full_2_limited(rgb2020));
        % convert to limit range and 12bit uint to suit the 'rgbwide2xyz'
        % function in MATLAB ver. > R2020b
    end

    % M1 (M1 = M3 * M2^-1)
    switch p.Results.non_linearity % M3
        case 'PQ'
            xyz = rgbwide2xyz(rgb2020,12,'ColorSpace','BT.2100',...
                'LinearizationFcn','PQ');
        case 'HLG'
            xyz = rgbwide2xyz(rgb2020,12,'ColorSpace','BT.2100',...
                'LinearizationFcn','HLG');
        case 'gamma'
            xyz = rgbwide2xyz(rgb2020,12,'ColorSpace','BT.2020');
        otherwise
            error('Unsupported Non-linearity!')
    end

    if strcmp(p.Results.target_gamut,'srgb') == true
        p.Results.target_gamut = 'bt709';
    end

    switch p.Results.target_gamut % M2^-1
        case 'bt709'
            rgb709 = xyz2rgb(xyz,'ColorSpace','srgb');
        case 'adobergb'
            rgb709 = xyz2rgb(xyz,'ColorSpace','adobe-rgb-1998');
        otherwise
            error('Unsupported Target Gamut!')
    end

    % hard clip OOG RGB values accroding to simple method (BT.2407 §2, RGB
    % values <0 or >1 are all clipped to boundary 0 or 1)
    rgb709_clipped = rgb709;

    % output the percentage of OOG pixels (FWGP) (our EXTENSION)
    oogPx = numel(rgb709_clipped(rgb709_clipped<0)) + numel(rgb709_clipped(rgb709_clipped>1));
    percentage = oogPx/numel(rgb709_clipped);
    %

    rgb709_clipped(rgb709_clipped<0) = 0;
    rgb709_clipped(rgb709_clipped>1) = 1;

    if p.Results.output_clipped709 == true
        varargout{1} = rgb709_clipped;
    end

    switch p.Results.target_gamut % M2
        case 'bt709'
            xyz_clipped = rgb2xyz(rgb709_clipped,'ColorSpace','srgb');
        case 'adobergb'
            xyz_clipped = rgb2xyz(rgb709_clipped,...
                'ColorSpace','adobe-rgb-1998');
        otherwise
            error('Unsupported Target Gamut!')
    end

    % PATH 2 (below path in paper's Fig. 9): XYZ after hard-chip
    % xyz_unclipped = xyz

    % FINAL STEP
    % Comparing original and clipped XYZ value using Euclidean diatance

    if strcmp(p.Results.compare_mode,'Yxy') == true
        p.Results.target_gamut = 'xy';
    end

    shape = size(xyz);
    if strcmp(p.Results.compare_mode,'xy') == true
        cal_xy = @(xyz)(cat(3,...
            xyz(:,:,1)./sum(xyz,3),xyz(:,:,1)./sum(xyz,3)));
        xy = cal_xy(xyz);
        xy_clipped = cal_xy(xyz_clipped);
    end

    distance = zeros(shape(1), shape(2));
    switch p.Results.compare_mode
        case 'XYZ'
            for i=1:shape(1)
                for j=1:shape(2)
                    distance(i,j) =...
                        sum((xyz(i,j,:)-xyz_clipped(i,j,:)).^2).^0.5;
                end
            end
        case 'xy'
            for i=1:shape(1)
                for j=1:shape(2)
                    distance(i,j) =...
                        sum((xy(i,j,:)-xy_clipped(i,j,:)).^2).^0.5;
                end
            end
        otherwise
            error('Unsupported Compare Mode Name!')
    end

    % norlmalize accroding to max posiable distance 0.234742440306215
    % (origin paper declare that it occurs when input green [0 1 0],
    % but we found it at [1 0.309803921568627 0.117647058823529] where
    % (XYZ(0.6573 0.3395, 0.0319), xy(0.6390 0.3300), B-R near R)
    % max distance is 0.275068397068084, by feeding a 4096*4096*3 CMS test
    % pattern containing all 256^3 possiable color combinations)
    degree = mean(distance(:))/0.275068397068084;
    if p.Results.output_oog_heatmap == true
        varargout{2} = distance/0.275068397068084; % which can later use as
        % imwrite(varargout{2}, trubo(60), 'name.jpg') in breakpoint
    end
