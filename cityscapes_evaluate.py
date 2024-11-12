from __future__ import absolute_import, division, print_function
import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from layers import disp_to_depth
from utils import readlines
from options import LiteMonoOptions
import datasets
import networks
import time
from thop import clever_format
from thop import profile
from PIL import Image
import numpy as np
import os


# 禁用 OpenCV 的多线程功能，将线程数设为 0
# 这样可以避免在多线程环境中出现资源竞争和性能不稳定的问题
# 通常在需要手动控制多线程或在多进程应用中使用 OpenCV 时会设置此参数
cv2.setNumThreads(0)

# 获取当前脚本文件的目录路径，并与 "splits" 拼接成完整路径,相当于获取..\Lite-Mono-main\splits
splits_dir = os.path.join(os.path.dirname(__file__), "splits")


# 主要是用于计算params以及Flops
# 此处的输入是encoder和decoder模型，以及一个张量x
def profile_once(encoder, decoder, x):
    """
    主要是用于计算params以及Flops
    此处的输入是encoder和decoder模型，以及一个张量x
    """
    # 提取输入张量 x 的第一个样本并增加一个维度,这可以确保 x_e 的形状符合 encoder 的输入需求，将其变成（1,channels,width,height）
    x_e = x[0, :, :, :].unsqueeze(0)
    # 使用 encoder 对 x_e 进行编码，得到编码结果 x_d
    x_d = encoder(x_e)
    # 使用 profile 函数计算 encoder 的浮点运算量 (FLOPs) 和参数数量
    flops_e, params_e = profile(encoder, inputs=(x_e,), verbose=False)
    flops_d, params_d = profile(decoder, inputs=(x_d,), verbose=False)
    # 将 encoder 和 decoder 的 FLOPs 与参数数量加总后格式化输出
    # clever_format 用于将大数字格式化为可读性更高的格式，如 1.234G
    flops, params = clever_format([flops_e + flops_d, params_e + params_d], "%.3f")
    # 分别格式化 encoder 的 FLOPs 和参数数量
    flops_e, params_e = clever_format([flops_e, params_e], "%.3f")
    # 分别格式化 decoder 的 FLOPs 和参数数量
    flops_d, params_d = clever_format([flops_d, params_d], "%.3f")
    # 返回整体和各部分的 FLOPs 与参数数量，分别包括总量和各模块的具体值
    return flops, params, flops_e, params_e, flops_d, params_d

def compute_errors(gt, pred):
    """计算预测深度和真实深度之间的误差指标。
    参数:
    gt: 真实深度值数组 (ground truth)。
    pred: 预测深度值数组。

    返回:
    abs_rel: 平均绝对相对误差。
    sq_rel: 平均平方相对误差。
    rmse: 均方根误差。
    rmse_log: 对数均方根误差。
    a1, a2, a3: 误差阈值指标 (accuracy)。
    """
    # 计算误差阈值指标 (accuracy)，在不同阈值下的预测准确率
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    # 计算均方根误差 (RMSE)
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    # 计算对数均方根误差 (RMSE log)
    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())
    # 计算平均绝对相对误差 (Absolute Relative Difference)
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    # 计算平均平方相对误差 (Squared Relative Difference)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    # 返回各个误差指标
    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3



def batch_post_process_disparity(l_disp, r_disp):
    """对视差图进行后处理，方法参考 Monodepthv1。
    通常来说不会启用这个模块
    参数:
    l_disp: 左视差图，形状为 (batch_size, height, width)。
    r_disp: 右视差图，形状为 (batch_size, height, width)。
    返回:
    经过后处理的视差图。
    """
    # 获取视差图的高度 h 和宽度 w
    _, h, w = l_disp.shape
    # 计算左右视差图的平均视差图
    m_disp = 0.5 * (l_disp + r_disp)
    # 创建一个从左到右的线性渐变掩码矩阵 l，用于控制左右视差图的加权
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    # 生成左侧掩码，逐渐将左视差图的影响减少到右视差图
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    # 生成右侧掩码，与左掩码相反
    r_mask = l_mask[:, :, ::-1]
    # 返回融合的视差图：左掩码控制左视差图，右掩码控制右视差图，中间区域使用平均视差图
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp



def evaluate(opt):
    """
    使用指定的测试集评估一个预训练模型
    options涉及内容
    --data_path
    --log_dir
    --model
    --height
    --width
    --load_weights_folder
    --eval_split
    """
    MIN_DEPTH = 1e-3  # 设置深度值的最小限制
    MAX_DEPTH = 80  # 设置深度值的最大限制

    # 检查是否有外部的视差文件需要评估（如果没有则进行模型加载和数据准备），city似乎提供了，如果不需要加载才进行接下来的操作
    if opt.ext_disp_to_eval is None:
        # 展开加载权重的文件夹路径，此处主要是处理路径，让其成为绝对路径，此处用到的args为load_weights_folder
        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)
        # 确认加载权重文件夹是否存在
        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)
        print("-> Loading weights from {}".format(opt.load_weights_folder))
        # 读取测试文件列表，后面一大串都是拼接路径用的，尤其注意splits_dir，opt.eval_split例如，如果 splits_dir 是 /data/dataset，而 opt.eval_split 是 test，那么最终路径就是 /data/dataset/test/test_files.txt
        # 替换文件在这进行更改
        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))

        # 定义编码器和解码器模型的权重文件路径
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")
        # 读取编码器和解码器的权重文件，两者是包含权重信息的实参，但没有架构
        encoder_dict = torch.load(encoder_path)
        decoder_dict = torch.load(decoder_path)

        # 初始化编码器和深度解码器。内部没有权重信息，只有架构信息，其权重均为初始化权重
        # 编码器自定义在这完成
        encoder = networks.LiteMono(model=opt.model,
                                    height=encoder_dict['height'],
                                    width=encoder_dict['width'])
        depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(3))

        # 修改主要是在这一块
        # frames_to_load = [0]
        # # 如果启用未来帧选项，则将未来帧索引 1 添加到帧列表
        # if opt.use_future_frame:
        #     frames_to_load.append(1)
        # # 遍历 -1 到 -1 - opt.num_matching_frames 的范围，步长为 -1，表示逐渐向过去的帧添加
        # # 比如，当 opt.num_matching_frames = 3 时，这段代码会依次添加 -1, -2, -3 表示的历史帧
        # for idx in range(-1, -1 - opt.num_matching_frames, -1):
        #     # 检查当前索引 idx 是否已在帧列表中
        #     # 避免重复添加相同的帧
        #     if idx not in frames_to_load:
        #         # 如果当前索引不在列表中，则将其添加
        #         frames_to_load.append(idx)
        # # 最终 frames_to_load 包含所有要加载的帧索引
        # # 如果 opt.use_future_frame = True 且 opt.num_matching_frames = 3，frames_to_load 将为 [0, 1, -1, -2, -3]

        dataset = datasets.CityscapesEvalDataset(opt.data_path,
                                                 filenames,
                                                 opt.height,  # 图像高度
                                                 opt.width,  # 图像宽度
                                                 [0],
                                                 4,
                                                 is_train=False)
        # 使用 DataLoader 按批次加载数据
        dataloader = DataLoader(dataset, 16, shuffle=False, num_workers=opt.num_workers, pin_memory=True, drop_last=False)
        # 读取初始化的权重（即空白信息）
        model_dict = encoder.state_dict()
        depth_model_dict = depth_decoder.state_dict()
        #比较键值对，核对将权重的数据转移到model_dict然后将其读取到encoder中，此时完成权重和架构的同时加载
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in depth_model_dict})
        # 将模型移到 GPU 上并设置为评估模式
        encoder.cuda()
        encoder.eval()
        depth_decoder.cuda()
        depth_decoder.eval()
        input_color = torch.ones(1, 3, opt.height, opt.width).cuda()
        flops, params, flops_e, params_e, flops_d, params_d = profile_once(encoder, depth_decoder, input_color)
        print("\n  " + ("flops: {0}, params: {1}, flops_e: {2}, params_e:{3}, flops_d:{4}, params_d:{5}").format(flops,
                                                                                                                 params,
                                                                                                                 flops_e,
                                                                                                                 params_e,
                                                                                                                 flops_d,
                                                                                                                 params_d) + "\n")
        # 初始化存储深度预测结果的列表
        pred_disps = []
        # 打印预测图像的尺寸
        print("-> Computing predictions with size {}x{}".format(opt.width, opt.height))
        # 禁用梯度计算，进入推理阶段
        with torch.no_grad():
            for data in dataloader:
                # 将当前批次的数据移动到 GPU 上
                input_color = data[("color", 0, 0)].cuda()
                # 如果启用了后处理，则在批次中添加翻转的图像，一般是不启用
                if opt.post_process:
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)
                # 计算编码器和解码器的计算量（FLOPS）和参数数量
                # flops, params, flops_e, params_e, flops_d, params_d = profile_once(encoder, depth_decoder, input_color)
                # 记录开始时间
                t1 = time_sync()
                # 前向传播，计算深度图，这里好像输出的是图像
                output = depth_decoder(encoder(input_color))
                # 记录结束时间
                t2 = time_sync()
                # 获取解码器的输出，并转换为深度图，我们需要的是固定比例下的图
                pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                # 将预测的深度图移到 CPU 并转换为 numpy 数组
                pred_disp = pred_disp.cpu()[:, 0].numpy()
                # 如果启用了后处理，则执行批量后处理
                if opt.post_process:
                    # 如果启用了后处理，获取图像数量（批量中的一半，因为之前进行了翻转拼接）
                    N = pred_disp.shape[0] // 2  # 预测视差图数组的前半部分和后半部分
                    # 调用 batch_post_process_disparity 函数，对视差图进行后处理
                    # pred_disp[:N] 为原始预测视差图，pred_disp[N:, :, ::-1] 为翻转后的视差图
                    pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])
                # 添加当前批次的预测深度图到结果列表
                pred_disps.append(pred_disp)
        # 合并所有批次的预测深度图为一个 numpy 数组
        pred_disps = np.concatenate(pred_disps)


    # 这里基本上用不到，因为要的是模型生成的深度图
    else:
        # 如果不进行计算，而是从文件中加载之前保存的预测结果
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        # 使用 numpy 加载存储的深度预测数据文件（即之前计算并保存的视差图）
        pred_disps = np.load(opt.ext_disp_to_eval)

        # 如果启用了 Eigen 基准的评估功能，将预测结果与基准数据集进行对齐，好像options里面压根没有这一条也就不启用了
        if opt.eval_eigen_to_benchmark:
            # 加载用于将 Eigen 数据集的图像 ID 映射到基准数据集 ID 的映射文件
            # 该文件包含 Eigen 测试集中图像与其他基准数据集中图像的对应关系
            eigen_to_benchmark_ids = np.load(
                os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))

            # 根据映射文件对预测结果进行重新排序，以确保它们与基准数据集的顺序一致
            pred_disps = pred_disps[eigen_to_benchmark_ids]


    if opt.save_pred_disps:
        # 构建输出文件路径，将预测结果保存在加载权重的文件夹中，并命名为 "disps_{eval_split}_split.npy"
        output_path = os.path.join(
            opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
        # 打印提示信息，显示将视差结果保存到的路径
        print("-> Saving predicted disparities to ", output_path)


    if opt.no_eval:
        print("-> Evaluation disabled. Done.")
        quit()

    if opt.eval_split == 'cityscapes':
        print('loading cityscapes gt depths individually due to their combined size!')
        # gt_depths = os.path.join(splits_dir, opt.eval_split, "gt_depths")
        gt_depths = '/data/mjc/AAAI/CS_NPY/gt_depths/'
    else:
        gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
        gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]
    print("-> Evaluating")
    print("   Mono evaluation - using median scaling")



    #正式开始计算
    errors = []
    ratios = []

    # 遍历所有预测的视差图
    for i in range(pred_disps.shape[0]):
        if opt.eval_split == 'cityscapes':
            # 加载对应编号的深度图，路径为'gt_depths'文件夹下对应编号的深度图文件
            gt_depth = np.load(os.path.join(gt_depths, str(i).zfill(3) + '_depth.npy'))
            # 获取深度图的高度和宽度
            gt_height, gt_width = gt_depth.shape[:2]
            # 裁剪地面真实深度数据，以去除与自车相关的部分（在数据加载过程中已经裁剪了输入图像）
            gt_height = int(round(gt_height * 0.75))  # 将高度裁剪为原高度的75%
            # 根据新的高度裁剪深度图，去除上方不需要的部分
            gt_depth = gt_depth[:gt_height]


        else:
            gt_depth = (1/gt_depths[i])
            gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]  # 获取第 i 张图的预测视差图
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))  # 将预测视差图调整为真实深度图的大小
        pred_depth = 1 / pred_disp  # 计算预测深度图（视差与深度的关系是深度 = 1 / 视差）
        # pred_depth = pred_disp  # 视情况而定是否需要1/视差

        if opt.eval_split == 'cityscapes':
            # 在评估 Cityscapes 数据集时，我们对图像进行中心裁剪，保留中间的 50%
            # 在数据预处理中，地面真实深度图的底部 25% 已经裁剪
            # 这里进一步裁剪图像的左右两侧和顶部
            # 对地面真实深度图进行裁剪，仅保留中间的 50%
            gt_depth = gt_depth[256:, 192:1856]

            # 对预测深度图也进行相同的裁剪，以保持尺寸一致
            pred_depth = pred_depth[256:, 192:1856]
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

        else:
            # 如果评估分割不是 "eigen"
            mask = gt_depth > 0  # 创建掩膜，筛选出有效的真实深度（大于 0）



        pred_depth = pred_depth[mask]  # 根据掩膜应用筛选，获取有效的预测深度
        gt_depth = gt_depth[mask]  # 根据掩膜应用筛选，获取有效的真实深度

        pred_depth *= opt.pred_depth_scale_factor  # 根据选项缩放预测深度

        if not opt.disable_median_scaling:
            # 如果没有禁用中位数缩放
            ratio = np.median(gt_depth) / np.median(pred_depth)  # 计算真实深度与预测深度的中位数比
            ratios.append(ratio)  # 将比率存储到列表中
            pred_depth *= ratio  # 根据比率调整预测深度

        # 限制预测深度在最小和最大深度范围内
        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH  # 小于最小深度的值设置为最小深度
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH  # 大于最大深度的值设置为最大深度
        errors.append(compute_errors(gt_depth, pred_depth))  # 计算并存储预测深度与真实深度的误差

    # 如果没有禁用中位数缩放
    if not opt.disable_median_scaling:
        ratios = np.array(ratios)  # 将比率列表转换为数组
        med = np.median(ratios)  # 计算比率的中位数
        # 打印缩放比率的中位数和标准差
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)  # 计算所有误差的平均值

    # 打印误差结果，展示各项指标：abs_rel、sq_rel、rmse、rmse_log、a1、a2、a3
    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")

    # 打印硬件信息（如 FLOPS 和参数量）
    print("\n  " + ("flops: {0}, params: {1}, flops_e: {2}, params_e:{3}, flops_d:{4}, params_d:{5}").format(flops,
                                                                                                             params,
                                                                                                             flops_e,
                                                                                                             params_e,
                                                                                                             flops_d,
                                                                                                             params_d))

    print("\n-> Done!")


if __name__ == "__main__":
    options = LiteMonoOptions()
    evaluate(options.parse())