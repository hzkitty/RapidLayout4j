package io.github.hzkitty.rapidlayout.utils.pre;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

public class DocLayoutPreProcess {
    private final int targetWidth;
    private final int targetHeight;

    public DocLayoutPreProcess(int width, int height) {
        this.targetWidth = width;
        this.targetHeight = height;
    }

    /**
     * 流程：1) BGR->RGB => 2) resize => 3) /255 => 4) permute => 5) expandDims
     */
    public float[][][][] call(Mat img) {
        if (img == null || img.empty()) {
            throw new IllegalArgumentException("图像为空");
        }

        // 1. BGR->RGB
        Mat rgbImg = new Mat();
        Imgproc.cvtColor(img, rgbImg, Imgproc.COLOR_BGR2RGB);

        // 2. resize
        Mat resized = new Mat();
        Imgproc.resize(rgbImg, resized, new Size(targetWidth, targetHeight));

        // 3. /255
        resized.convertTo(resized, CvType.CV_32FC3, 1.0 / 255.0);

        // 4. permute
        float[][][] permuted = permute(resized);

        // 5. expandDims
        return expandDims(permuted);
    }

    private float[][][] permute(Mat img) {
        int height = img.rows();
        int width = img.cols();
        int channels = img.channels();
        float[][][] output = new float[channels][height][width];
        float[] buffer = new float[height * width * channels];
        img.get(0, 0, buffer);

        int index = 0;
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                for (int c = 0; c < channels; c++) {
                    output[c][h][w] = buffer[index++];
                }
            }
        }
        return output;
    }

    private float[][][][] expandDims(float[][][] input) {
        int c = input.length;
        int h = input[0].length;
        int w = input[0][0].length;
        float[][][][] output = new float[1][c][h][w];
        for (int ci = 0; ci < c; ci++) {
            for (int hi = 0; hi < h; hi++) {
                System.arraycopy(input[ci][hi], 0, output[0][ci][hi], 0, w);
            }
        }
        return output;
    }
}