package io.github.hzkitty.rapid_layout.utils.post;

import io.github.hzkitty.rapid_layout.entity.Triple;
import org.opencv.core.Size;

import java.util.ArrayList;
import java.util.List;

public class DocLayoutPostProcess {
    private final List<String> labels;
    private final double confThreshold;
    private final double iouThreshold;

    private int imgHeight, imgWidth;
    private int inputHeight, inputWidth;

    public DocLayoutPostProcess(List<String> labels, double confThres, double iouThres) {
        this.labels = labels;
        this.confThreshold = confThres;
        this.iouThreshold = iouThres;
    }

    public Triple<List<float[]>, List<Float>, List<String>> call(float[][][][] output, Size oriImgShape, int[] imgShape) {
        // 设置原始图像和输入图像的尺寸
        this.imgHeight = (int) oriImgShape.height;
        this.imgWidth = (int) oriImgShape.width;
        this.inputHeight = imgShape[0];
        this.inputWidth = imgShape[1];

        float[][] outputBatch = squeeze2D(output[0]);

        // 提取框坐标、置信度和类别ID
        List<float[]> boxesList = new ArrayList<>();
        List<Float> confidencesList = new ArrayList<>();
        List<Integer> classIdsList = new ArrayList<>();

        for (int i = 0; i < outputBatch.length; i++) {
            float[] detection = outputBatch[i];
            float confidence = detection[detection.length - 2];
            int classId = (int) detection[detection.length - 1];
            if (confidence > this.confThreshold) {
                // 提取框坐标
                float[] box = new float[detection.length - 2];
                System.arraycopy(detection, 0, box, 0, detection.length - 2);
                boxesList.add(box);
                confidencesList.add(confidence);
                classIdsList.add(classId);
            }
        }

        // 将列表转换为数组进行后续处理
        float[] confidences = new float[confidencesList.size()];
        for (int i = 0; i < confidencesList.size(); i++) {
            confidences[i] = confidencesList.get(i);
        }
        int[] classIds = classIdsList.stream().mapToInt(Integer::intValue).toArray();

        // 调整框的尺寸到原始图像尺寸
        boxesList = rescaleBoxes(boxesList, this.inputWidth, this.inputHeight, this.imgWidth, this.imgHeight);

        // 根据类别ID获取标签
        List<String> labelsList = new ArrayList<>();
        for (int classId : classIds) {
            if (classId >= 0 && classId < this.labels.size()) {
                labelsList.add(this.labels.get(classId));
            } else {
                labelsList.add("Unknown");
            }
        }

        return Triple.of(boxesList, confidencesList, labelsList);
    }

    private float[][] squeeze2D(float[][][] arr) {
        return arr[0];
    }

    /**
     * 调整检测框的尺寸到原始图像尺寸
     *
     * @param boxes       原始检测框
     * @param inputWidth  输入图像的宽度
     * @param inputHeight 输入图像的高度
     * @param imgWidth    原始图像的宽度
     * @param imgHeight   原始图像的高度
     * @return 调整后的检测框
     */
    private List<float[]> rescaleBoxes(List<float[]> boxes, int inputWidth, int inputHeight, int imgWidth, int imgHeight) {
        float scaleX = (float) imgWidth / inputWidth;
        float scaleY = (float) imgHeight / inputHeight;

        List<float[]> rescaledBoxes = new ArrayList<>(boxes.size());
        for (float[] box : boxes) {
            // 框的格式为 [x, y, w, h]
            float x = box[0] * scaleX; // x
            float y = box[1] * scaleY; // y
            float w = box[2] * scaleX; // w
            float h = box[3] * scaleY; // h
            rescaledBoxes.add(new float[]{x, y, w, h});
        }
        return rescaledBoxes;
    }
}