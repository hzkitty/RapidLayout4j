package io.github.hzkitty.rapidlayout.utils.pre;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

public class PPPreProcess {
    private final Size imgSize;
    // 均值和标准差 (RGB 顺序)
    private final Scalar mean = new Scalar(0.485, 0.456, 0.406);
    private final Scalar std = new Scalar(0.229, 0.224, 0.225);
    private final double scale = 1.0 / 255.0;  // 归一化因子

    public PPPreProcess(Size imgSize) {
        this.imgSize = imgSize;
    }

    /**
     * 整体流程：1) resize  2) normalize 3) permute  4) expandDims
     *
     * @param img OpenCV Mat 输入（BGR 格式）
     * @return 四维 float 数组 [1, C, H, W]
     */
    public float[][][][] call(Mat img) {
        if (img == null || img.empty()) {
            throw new IllegalArgumentException("传入的图像为空");
        }
        // 1. 调整尺寸
        Mat resized = resize(img);

        // 2. 归一化 ( (pixel*scale - mean)/std ), 这里的 mean/std 按 RGB 顺序
        Mat normalized = normalize(resized);

        // 3. 维度变换 (H, W, C) => (C, H, W)
        float[][][] permuted = permute(normalized);

        // 4. 扩展维度 => [1, C, H, W]
        return expandDims(permuted);
    }

    private Mat resize(Mat img) {
        Mat dst = new Mat();
        Imgproc.resize(img, dst, imgSize);
        return dst;
    }

    private Mat normalize(Mat img) {
        img.convertTo(img, CvType.CV_32FC3, scale);
        Core.subtract(img, mean, img);
        Core.divide(img, std, img);
        return img;
    }

    /**
     * 将图像从 (H,W,C) 转换为 (C,H,W) 的三维 float 数组
     */
    private float[][][] permute(Mat img) {
        int h = img.rows();
        int w = img.cols();
        int c = img.channels();
        float[][][] output = new float[c][h][w];

        for (int row = 0; row < h; row++) {
            for (int col = 0; col < w; col++) {
                double[] pixel = img.get(row, col);  // size = c
                for (int ch = 0; ch < c; ch++) {
                    output[ch][row][col] = (float) pixel[ch];
                }
            }
        }
        return output;
    }

    /**
     * 扩展一个 batch 维度 => [1, C, H, W]
     */
    private float[][][][] expandDims(float[][][] permuted) {
        int c = permuted.length;
        int h = permuted[0].length;
        int w = permuted[0][0].length;

        float[][][][] out = new float[1][c][h][w];
        for (int cc = 0; cc < c; cc++) {
            for (int hh = 0; hh < h; hh++) {
                System.arraycopy(permuted[cc][hh], 0, out[0][cc][hh], 0, w);
            }
        }
        return out;
    }
}
