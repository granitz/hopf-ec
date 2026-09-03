function out = build_normative_sc(cfg)
%BUILD_NORMATIVE_SC  ROI-to-ROI structural connectivity from the dTOR-985 connectome.
%
%   out = build_normative_sc(cfg)
%
%   MATLAB companion of hopfec/sc/normative.py (same method, same outputs).  Paths are
%   given in the cfg struct (no hard-coded locations), e.g.
%
%     cfg.atlas_file      = fullfile(proj,'atlas','Schaefer...Tian...MNI152NLin2009cAsym_2mm.nii.gz');
%     cfg.labels_file     = fullfile(proj,'atlas','SchaeferTian116_labels.tsv');  % optional
%     cfg.connectome_file = fullfile(proj,'data','dTOR_fibers_vox_2_mm.mat');
%     cfg.output_file     = fullfile(proj,'out','SC_normative.mat');
%     cfg.fiber_grid      = 'RAS';   % 'RAS' (original script convention) or 'LAS' (FSL/SPM storage)
%     cfg.count_mode      = 'touched';  % 'touched' (original) | 'endpoints'
%
%   Method (identical to build_roi_to_roi_corrected.m):
%     fibers_vox holds 1-indexed voxel coordinates in the 91x109x91 MNI152 2 mm grid.  Each point
%     is mapped to the atlas grid through the two affines (fiber voxel -> mm -> atlas voxel), so any
%     atlas grid works.  A streamline touching parcels {p1..pk} adds 1 to every pair (and 1/length to
%     the length-weighted matrix).  Matrices are then volume-normalised (sqrt(vol_i*vol_j)).
%
%   Outputs are saved to cfg.output_file (-v7.3) with the same variable names as the original
%   script plus CSV copies next to it.

% ---------------------------------------------------------------- settings
if ~isfield(cfg,'fiber_grid'),  cfg.fiber_grid = 'RAS'; end
if ~isfield(cfg,'count_mode'),  cfg.count_mode = 'touched'; end
if ~isfield(cfg,'length_mode'), cfg.length_mode = 'npoints'; end
if ~isfield(cfg,'chunk_size'),  cfg.chunk_size = 200000; end
if ~isfield(cfg,'labels_file'), cfg.labels_file = ''; end
voxel_size_mm = 2;
switch upper(cfg.fiber_grid)
    case 'RAS', A_fiber = [2 0 0 -90; 0 2 0 -126; 0 0 2 -72; 0 0 0 1];
    case 'LAS', A_fiber = [-2 0 0 90; 0 2 0 -126; 0 0 2 -72; 0 0 0 1];
    otherwise
        if isnumeric(cfg.fiber_grid) && isequal(size(cfg.fiber_grid),[4 4])
            A_fiber = cfg.fiber_grid;
        else
            error('cfg.fiber_grid must be ''RAS'', ''LAS'' or a 4x4 affine (0-indexed voxel -> mm)');
        end
end

% ---------------------------------------------------------------- atlas
info  = niftiinfo(cfg.atlas_file);
atlas = round(double(niftiread(cfg.atlas_file)));
atlas(atlas < 0) = 0;
A_atlas = info.Transform.T';                       % 0-indexed voxel -> mm (row = [x y z 1])
labels  = unique(atlas(:)); labels(labels == 0) = [];
nROIs   = numel(labels);
atlas_dim = size(atlas);
fprintf('Atlas: %s  (%d x %d x %d, %d parcels)\n', cfg.atlas_file, atlas_dim, nROIs);
label_map = zeros(max(labels), 1);
label_map(labels) = 1:nROIs;
roi_vol = accumarray(label_map(atlas(atlas > 0)), 1, [nROIs 1]);

roi_names = strings(nROIs,1); roi_hemisphere = strings(nROIs,1); roi_type = strings(nROIs,1); roi_network = strings(nROIs,1);
for k = 1:nROIs, roi_names(k) = sprintf('roi_%d', labels(k)); end
if ~isempty(cfg.labels_file) && exist(cfg.labels_file, 'file')
    T = readtable(cfg.labels_file, 'FileType', 'text', 'Delimiter', '\t');
    idcol = find(ismember(lower(T.Properties.VariableNames), {'index','id','label','value'}), 1);
    namecol = find(ismember(lower(T.Properties.VariableNames), {'name','region','label_name'}), 1);
    if ~isempty(idcol) && ~isempty(namecol)
        ids = double(T{:, idcol}); nm = string(T{:, namecol});
        for k = 1:nROIs
            j = find(ids == labels(k), 1);
            if ~isempty(j), roi_names(k) = nm(j); end
        end
    end
    hc = find(ismember(lower(T.Properties.VariableNames), {'hemisphere','hemi'}), 1);
    if ~isempty(hc), roi_hemisphere = upper(extractBefore(string(T{:, hc}) + " ", 2)); end
    tc = find(ismember(lower(T.Properties.VariableNames), {'type'}), 1);
    if ~isempty(tc), roi_type = string(T{:, tc}); end
    nc = find(ismember(lower(T.Properties.VariableNames), {'network'}), 1);
    if ~isempty(nc), roi_network = string(T{:, nc}); end
end
if ~any(strlength(roi_hemisphere))
    for k = 1:nROIs
        n = roi_names(k);
        if ~isempty(regexp(n, '(^|[_\-])(LH|lh|L|Left|left)([_\-]|$)', 'once')), roi_hemisphere(k) = "L";
        elseif ~isempty(regexp(n, '(^|[_\-])(RH|rh|R|Right|right)([_\-]|$)', 'once')), roi_hemisphere(k) = "R"; end
    end
end

% fiber voxel (1-indexed) -> atlas voxel (0-indexed) : M = inv(A_atlas) * A_fiber * shift(-1)
shift = eye(4); shift(1:3,4) = -1;
M = A_atlas \ (A_fiber * shift);
fprintf('Fiber grid %s; fiber->atlas offset (0-indexed voxels): [%.3f %.3f %.3f]\n', upper(string(cfg.fiber_grid)), M(1:3,4));

% ---------------------------------------------------------------- fibers
fprintf('Loading connectome %s ... ', cfg.connectome_file);
S = load(cfg.connectome_file);
fn = fieldnames(S);
if isfield(S, 'fibers_vox'), fibers_vox = S.fibers_vox; else, fibers_vox = S.(fn{1}); end
clear S
nFibers = numel(fibers_vox);
fprintf('%d streamlines.\n', nFibers);

conn_matrix = zeros(nROIs); conn_matrix_len = zeros(nROIs);
fiber_lengths = zeros(nFibers, 1);
n_connecting = 0;
tic
for start = 1:cfg.chunk_size:nFibers
    stop = min(start + cfg.chunk_size - 1, nFibers);
    chunk = fibers_vox(start:stop);
    npts = cellfun(@(c) size(c, 1) * (size(c, 2) == 3) + size(c, 2) * (size(c, 1) == 3 && size(c, 2) ~= 3), chunk);
    pts = zeros(sum(npts), 3);
    pos = 0;
    for i = 1:numel(chunk)
        c = double(chunk{i});
        if size(c, 1) == 3 && size(c, 2) ~= 3, c = c'; end
        pts(pos+1:pos+size(c,1), :) = c;
        pos = pos + size(c, 1);
    end
    sid = repelem((1:numel(chunk))', npts);
    if strcmpi(cfg.length_mode, 'polyline')
        world = pts * A_fiber(1:3,1:3)';
        seg = sqrt(sum(diff(world, 1, 1).^2, 2));
        cs = [0; cumsum(seg)];
        offs = [0; cumsum(npts)];
        L = max(cs(max(offs(2:end), 1)) - cs(offs(1:end-1) + 1), voxel_size_mm);
    else
        L = npts * voxel_size_mm;
    end
    fiber_lengths(start:stop) = L;
    v = round(pts * M(1:3,1:3)' + M(1:3,4)') + 1;          % 1-indexed atlas voxel
    inside = all(v >= 1, 2) & v(:,1) <= atlas_dim(1) & v(:,2) <= atlas_dim(2) & v(:,3) <= atlas_dim(3);
    lab = zeros(size(pts, 1), 1);
    lab(inside) = atlas(sub2ind(atlas_dim, v(inside,1), v(inside,2), v(inside,3)));
    hit = lab > 0;
    if strcmpi(cfg.count_mode, 'endpoints')
        offs = [0; cumsum(npts)];
        first = lab(offs(1:end-1) + 1); last = lab(max(offs(2:end), 1));
        ok = first > 0 & last > 0 & first ~= last;
        ia = label_map(first(ok)); ib = label_map(last(ok)); w = 1 ./ L(ok);
        n_connecting = n_connecting + nnz(ok);
        conn_matrix = conn_matrix + accumarray([ia ib], 1, [nROIs nROIs]) + accumarray([ib ia], 1, [nROIs nROIs]);
        conn_matrix_len = conn_matrix_len + accumarray([ia ib], w, [nROIs nROIs]) + accumarray([ib ia], w, [nROIs nROIs]);
    else
        pairs = unique([sid(hit) label_map(lab(hit))], 'rows');   % (streamline, ROI) incidence
        Minc = sparse(pairs(:,1), pairs(:,2), 1, numel(chunk), nROIs);
        n_connecting = n_connecting + nnz(sum(Minc, 2) >= 2);
        conn_matrix = conn_matrix + full(Minc' * Minc);
        conn_matrix_len = conn_matrix_len + full(Minc' * spdiags(1 ./ L, 0, numel(chunk), numel(chunk)) * Minc);
    end
    fprintf('  %d / %d streamlines (%.1f%%, %.0fs)\n', stop, nFibers, 100 * stop / nFibers, toc);
end
conn_matrix(1:nROIs+1:end) = 0;
conn_matrix_len(1:nROIs+1:end) = 0;

% ---------------------------------------------------------------- normalise
gv = sqrt(roi_vol * roi_vol'); gv(gv == 0) = Inf;
conn_matrix_volnorm  = conn_matrix ./ gv;
conn_matrix_lencorr  = conn_matrix_len;
conn_matrix_combined = conn_matrix_len ./ gv;
conn_matrix_volnorm(1:nROIs+1:end) = 0;
conn_matrix_combined(1:nROIs+1:end) = 0;

% ---------------------------------------------------------------- save
fiber_grid = upper(string(cfg.fiber_grid)); atlas_file = cfg.atlas_file; fiber_affine = A_fiber; %#ok<NASGU>
[outdir, outname] = fileparts(cfg.output_file);
if ~isempty(outdir) && ~exist(outdir, 'dir'), mkdir(outdir); end
save(cfg.output_file, 'conn_matrix', 'conn_matrix_volnorm', 'conn_matrix_lencorr', 'conn_matrix_combined', ...
    'labels', 'nROIs', 'roi_vol', 'roi_names', 'roi_hemisphere', 'roi_type', 'roi_network', ...
    'fiber_lengths', 'atlas_file', 'fiber_grid', 'fiber_affine', 'n_connecting', '-v7.3');
writematrix(conn_matrix,          fullfile(outdir, [outname '_count.csv']));
writematrix(conn_matrix_combined, fullfile(outdir, [outname '_combined.csv']));
writematrix(conn_matrix_volnorm,  fullfile(outdir, [outname '_volnorm.csv']));
writematrix(conn_matrix_lencorr,  fullfile(outdir, [outname '_lencorr.csv']));
fprintf('Saved %s\n', cfg.output_file);

% ---------------------------------------------------------------- sanity checks
fprintf('\n--- Sanity checks ---\n');
fprintf('Streamlines: %d, connecting >= 2 parcels: %d\n', nFibers, n_connecting);
fprintf('Total connections (upper triangle): %d\n', sum(conn_matrix(triu(true(nROIs), 1))));
fprintf('Density: %.4f   Max edge: %d   Empty nodes: %d   Symmetric: %d\n', ...
    nnz(conn_matrix) / (nROIs * (nROIs - 1)), max(conn_matrix(:)), sum(sum(conn_matrix, 2) == 0), isequal(conn_matrix, conn_matrix'));
lh = find(roi_hemisphere == "L"); rh = find(roi_hemisphere == "R");
if ~isempty(lh) && ~isempty(rh)
    lh_sum = mean(sum(conn_matrix(lh, :), 2)); rh_sum = mean(sum(conn_matrix(rh, :), 2));
    fprintf('Mean row-sum L: %.1f  R: %.1f  L/R ratio (target ~1): %.3f\n', lh_sum, rh_sum, lh_sum / rh_sum);
end
out = struct('conn_matrix', conn_matrix, 'conn_matrix_volnorm', conn_matrix_volnorm, 'conn_matrix_lencorr', conn_matrix_lencorr, ...
    'conn_matrix_combined', conn_matrix_combined, 'labels', labels, 'roi_vol', roi_vol, 'roi_names', roi_names, ...
    'roi_hemisphere', roi_hemisphere, 'fiber_lengths', fiber_lengths, 'n_connecting', n_connecting, 'fiber_grid', fiber_grid);
end
