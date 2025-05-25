package io.github.hzkitty.rapidlayout.utils.post;

import io.github.hzkitty.rapidlayout.entity.Triple;
import org.opencv.core.Size;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class YOLOv8PostProcess {
    private final List<String> labels;
    private final float confThreshold;
    private final float iouThreshold;

    private int imgHeight, imgWidth;
    private int inputHeight, inputWidth;

    public YOLOv8PostProcess(List<String> labels, float confThres, float iouThres) {
        this.labels = labels;
        this.confThreshold = confThres;
        this.iouThreshold = iouThres;
    }

    /**
     * 转置二维 float 数组，类似于 numpy 的 .T 操作。
     *
     * @param matrix 原始二维数组
     * @return 转置后的二维数组
     */
    private float[][] transpose(float[][] matrix) {
        if (matrix.length == 0) return new float[0][0];
        int rows = matrix.length;
        int cols = matrix[0].length;
        float[][] transposed = new float[cols][rows];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                transposed[j][i] = matrix[i][j];
            }
        }
        return transposed;
    }

    /**
     * 处理模型输出，筛选出有效的检测框、置信度和标签。
     *
     * @param output        模型的输出，四维数组
     * @param oriImgShape   原始图像的尺寸 (height, width)
     * @param imgShape      输入图像的尺寸 (height, width)
     * @return 包含检测框、置信度和标签的 Triple 对象
     */
    public Triple<List<float[]>, List<Float>, List<String>> call(float[][][][] output, Size oriImgShape, int[] imgShape) {
        // 设置原始图像和输入图像的尺寸
        this.imgHeight = (int) oriImgShape.height;
        this.imgWidth = (int) oriImgShape.width;
        this.inputHeight = imgShape[0];
        this.inputWidth = imgShape[1];

        // 取第一个 batch 和第一个通道
        float[][] outputBatch = output[0][0];

        // 转置操作，类似于 np.squeeze(output[0]).T
        float[][] predictions = transpose(outputBatch);

        // 提取框坐标、置信度和类别ID
        List<float[]> boxesList = new ArrayList<>();
        List<Float> confidencesList = new ArrayList<>();
        List<Integer> classIdsList = new ArrayList<>();

        for (int i = 0; i < predictions.length; i++) {
            float[] detection = predictions[i];
            float maxScore = -Float.MAX_VALUE;
            int maxClassId = -1;
            for (int j = 4; j < detection.length; j++) {
                if (detection[j] > maxScore) {
                    maxScore = detection[j];
                    maxClassId = j - 4; // 类别ID从0开始
                }
            }

            if (maxScore > this.confThreshold) {
                // 提取框坐标 [x, y, w, h]
                boxesList.add(detection);
                confidencesList.add(maxScore);
                classIdsList.add(maxClassId);
            }
        }

        // 如果没有检测到任何目标，返回空列表
        if (confidencesList.isEmpty()) {
            return Triple.of(new ArrayList<>(), new ArrayList<>(), new ArrayList<>());
        }

        // 将 boxesList 转换为 float[][] 数组
        // 调整框的尺寸到原始图像尺寸，并转换为 List<float[]>
        float[][] rescaledBoxesList = extractBoxes(boxesList.toArray(new float[0][]));

        // 执行多类别的非极大值抑制（NMS）以抑制重叠的检测框
        List<Integer> keepIndices = multiclassNms(Arrays.stream(rescaledBoxesList)
                .collect(Collectors.toList()), confidencesList, classIdsList, this.iouThreshold);

        // 根据保留的索引获取最终的检测结果
        List<float[]> finalBoxes = new ArrayList<>();
        List<Float> finalScores = new ArrayList<>();
        List<String> finalLabels = new ArrayList<>();

        for (int idx : keepIndices) {
            finalBoxes.add(rescaledBoxesList[idx]);
            finalScores.add(confidencesList.get(idx));
            int classId = classIdsList.get(idx);
            if (classId >= 0 && classId < this.labels.size()) {
                finalLabels.add(this.labels.get(classId));
            } else {
                finalLabels.add("Unknown");
            }
        }

        return Triple.of(finalBoxes, finalScores, finalLabels);
    }

    /**
     * 提取预测结果中的框坐标，并进行缩放和格式转换。
     *
     * @param predictions 模型预测结果，二维数组，每行包含 [x, y, w, h, ...]
     * @return 缩放后的框坐标，转换为 [x1, y1, x2, y2] 格式
     */
    private float[][] extractBoxes(float[][] predictions) {
        // 提取框坐标 [x, y, w, h]
        float[][] boxes = Arrays.stream(predictions)
                .map(pred -> Arrays.copyOfRange(pred, 0, 4))
                .toArray(float[][]::new);

        // 缩放框到原始图像尺寸
        boxes = rescaleBoxes(boxes, this.inputWidth, this.inputHeight, this.imgWidth, this.imgHeight);

        // 转换为 [x1, y1, x2, y2] 格式
        boxes = xywh2xyxy(boxes);

        return boxes;
    }

    private float[][] rescaleBoxes(float[][] boxes, int inputWidth, int inputHeight, int imgWidth, int imgHeight) {
        float scaleX = (float) imgWidth / inputWidth;
        float scaleY = (float) imgHeight / inputHeight;

        float[][] rescaledBoxes = new float[boxes.length][boxes[0].length];
        for (int i = 0; i < boxes.length; i++) {
            // 假设框的格式为 [x, y, w, h]
            rescaledBoxes[i][0] = boxes[i][0] * scaleX; // x
            rescaledBoxes[i][1] = boxes[i][1] * scaleY; // y
            rescaledBoxes[i][2] = boxes[i][2] * scaleX; // w
            rescaledBoxes[i][3] = boxes[i][3] * scaleY; // h
            // 如果框的格式不同，请根据实际情况调整
        }
        return rescaledBoxes;
    }

    /**
     * 缩放检测框的尺寸到原始图像尺寸。
     *
     * @param boxes        原始检测框，格式为 [x, y, w, h]
     * @param inputWidth   输入图像的宽度
     * @param inputHeight  输入图像的高度
     * @param imgWidth     原始图像的宽度
     * @param imgHeight    原始图像的高度
     * @return 缩放后的检测框，格式为 [x, y, w, h]
     */
    private List<float[]> rescaleBoxes(List<float[]> boxes, int inputWidth, int inputHeight, int imgWidth, int imgHeight) {
        float scaleX = (float) imgWidth / inputWidth;
        float scaleY = (float) imgHeight / inputHeight;

        List<float[]> rescaledBoxes = new ArrayList<>(boxes.size());
        for (float[] box : boxes) {
            // 假设框的格式为 [x, y, w, h]
            float x = box[0] * scaleX; // x
            float y = box[1] * scaleY; // y
            float w = box[2] * scaleX; // w
            float h = box[3] * scaleY; // h
            rescaledBoxes.add(new float[]{x, y, w, h});
        }
        return rescaledBoxes;
    }

    /**
     * 执行多类别的非极大值抑制（NMS），以抑制重叠的检测框。
     *
     * @param boxes        缩放后的检测框列表，格式为 [x1, y1, x2, y2]
     * @param scores       每个检测框的置信度分数
     * @param classIds     每个检测框的类别ID
     * @param iouThreshold IoU 阈值
     * @return 保留的检测框索引列表
     */
    private List<Integer> multiclassNms(List<float[]> boxes, List<Float> scores, List<Integer> classIds, float iouThreshold) {
        List<Integer> keepBoxes = new ArrayList<>();
        List<Integer> uniqueClassIds = classIds.stream().distinct().collect(Collectors.toList());

        for (Integer classId : uniqueClassIds) {
            // 获取当前类别的所有检测框索引
            List<Integer> classIndices = new ArrayList<>();
            for (int i = 0; i < classIds.size(); i++) {
                if (classIds.get(i).equals(classId)) {
                    classIndices.add(i);
                }
            }

            // 获取当前类别的检测框和分数
            List<float[]> classBoxes = classIndices.stream().map(boxes::get).collect(Collectors.toList());
            List<Float> classScores = classIndices.stream().map(scores::get).collect(Collectors.toList());

            // 执行 NMS
            List<Integer> classKeep = nms(classBoxes, classScores, iouThreshold);

            // 将保留的框索引添加到最终列表
            for (Integer idx : classKeep) {
                keepBoxes.add(classIndices.get(idx));
            }
        }

        return keepBoxes;
    }

    /**
     * 执行非极大值抑制（NMS），以抑制重叠的检测框。
     *
     * @param boxes        当前类别的检测框列表，格式为 [x, y, w, h]
     * @param scores       当前类别的检测框分数
     * @param iouThreshold IoU 阈值
     * @return 保留的检测框索引列表
     */
    private List<Integer> nms(List<float[]> boxes, List<Float> scores, float iouThreshold) {
        // 创建一个索引列表
        List<Integer> indices = IntStream.range(0, scores.size()).boxed().collect(Collectors.toList());

        // 按分数降序排序索引
        indices.sort((i1, i2) -> Float.compare(scores.get(i2), scores.get(i1)));

        List<Integer> keepBoxes = new ArrayList<>();

        while (!indices.isEmpty()) {
            // 选择分数最高的框
            int current = indices.get(0);
            keepBoxes.add(current);

            if (indices.size() == 1) {
                break;
            }

            // 获取当前框
            float[] currentBox = boxes.get(current);

            // 创建一个新列表来存储剩余的框
            List<Integer> remaining = new ArrayList<>();

            for (int i = 1; i < indices.size(); i++) {
                int idx = indices.get(i);
                float[] otherBox = boxes.get(idx);
                float iou = computeIou(currentBox, otherBox);
                if (iou < iouThreshold) {
                    remaining.add(idx);
                }
            }

            // 更新索引列表
            indices = remaining;
        }

        return keepBoxes;
    }

    /**
     * 计算两个框的 IoU（交并比）。
     *
     * @param box1 第一个检测框，格式为 [x1, y1, x2, y2]
     * @param box2 第二个检测框，格式为 [x1, y1, x2, y2]
     * @return IoU 值
     */
    private float computeIou(float[] box1, float[] box2) {
        // 计算交集坐标
        float xmin = Math.max(box1[0], box2[0]);
        float ymin = Math.max(box1[1], box2[1]);
        float xmax = Math.min(box1[2], box2[2]);
        float ymax = Math.min(box1[3], box2[3]);

        // 计算交集面积
        float intersectionArea = Math.max(0, xmax - xmin) * Math.max(0, ymax - ymin);

        // 计算两个框的面积
        float box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1]);
        float box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1]);

        // 计算并集面积
        float unionArea = box1Area + box2Area - intersectionArea;

        // 计算 IoU
        return unionArea > 0 ? intersectionArea / unionArea : 0;
    }

    /**
     * 将 [x, y, w, h] 格式的框转换为 [x1, y1, x2, y2] 格式。
     *
     * @param boxes 原始框，格式为 [x, y, w, h]
     * @return 转换后的框，格式为 [x1, y1, x2, y2]
     */
    private float[][] xywh2xyxy(float[][] boxes) {
        float[][] convertedBoxes = new float[boxes.length][4];
        for (int i = 0; i < boxes.length; i++) {
            float x = boxes[i][0];
            float y = boxes[i][1];
            float w = boxes[i][2];
            float h = boxes[i][3];
            convertedBoxes[i][0] = x - w / 2; // x1
            convertedBoxes[i][1] = y - h / 2; // y1
            convertedBoxes[i][2] = x + w / 2; // x2
            convertedBoxes[i][3] = y + h / 2; // y2
        }
        return convertedBoxes;
    }

}