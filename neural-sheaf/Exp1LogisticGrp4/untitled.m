% 设置主路径
root_dir = 'D:\脑网络\neurocon\neurocon';
subfolders = dir(fullfile(root_dir, 'sub-*'));

count_success = 0;  % ✅ 初始化计数器

for k = 1:length(subfolders)
    sub_name = subfolders(k).name;
    sub_path = fullfile(root_dir, sub_name);

    mat_files = dir(fullfile(sub_path, '*AAL116*_timeseries.mat'));
    if isempty(mat_files)
        fprintf('⚠️ 跳过 %s：未找到符合的 .mat 文件\n', sub_name);
        continue;
    end

    mat_file_path = fullfile(sub_path, mat_files(1).name);
    fprintf('✅ 正在处理: %s\n', mat_file_path);
    mat = load(mat_file_path);

    var_names = fieldnames(mat);
    data_loaded = false;
    for vi = 1:length(var_names)
        data_candidate = mat.(var_names{vi});
        if isnumeric(data_candidate) && size(data_candidate, 2) == 116
            TSs = data_candidate;
            data_loaded = true;
            break;
        end
    end

    if ~data_loaded
        fprintf('❌ 跳过 %s：未找到 116 列的时间序列数据\n', sub_name);
        continue;
    end

    embedDims = repmat(3, 1, size(TSs,2));
    embedLags = repmat(1, 1, size(TSs,2));
    epsilonPercents = exp(linspace(log(0.001), log(1), 33))';
    eliminateSpan = zeros(size(TSs,2), 1);
    option = struct('skipCInd2', true);

    [CInd, ~, ~, ~, ~, ~, ~, ~, ~, ~, GlobalDelta] = ...
        ACX(TSs, embedDims, embedLags, epsilonPercents, eliminateSpan, option);

    save_path = fullfile(sub_path, [sub_name '_causality.mat']);
    save(save_path, 'CInd', 'GlobalDelta');
    fprintf('💾 保存完成: %s\n\n', save_path);

    count_success = count_success + 1;  % ✅ 成功计数 +1
end

% ✅ 统计总结
fprintf('🎯 共成功处理并保存了 %d 个被试数据\n', count_success);
disp('✅ 所有子文件夹处理完成！');

