package io.github.hzkitty.rapid_layout.utils.post;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import io.github.hzkitty.rapid_layout.entity.Triple;
import org.opencv.core.Size;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * 后处理预测结果，包括置信度过滤、NMS（非极大值抑制）等操作。
 */
public class PPPostProcess {
    private List<String> labels;
    private List<Integer> strides;
    private float confThresh;
    private float iouThresh;
    private int nmsTopK;
    private int keepTopK;

    /**
     * 构造函数
     *
     * @param labels    类别标签
     * @param confThresh 置信度阈值，默认0.4
     * @param iouThresh  IOU阈值，默认0.5
     */
    public PPPostProcess(List<String> labels, float confThresh, float iouThresh) {
        this.labels = labels;
        this.strides = Arrays.asList(8, 16, 32, 64);
        this.confThresh = confThresh;
        this.iouThresh = iouThresh;
        this.nmsTopK = 1000;
        this.keepTopK = 100;
    }

    /**
     * 主调用方法，进行后处理
     *
     * @param manager   NDManager 实例
     * @param oriShape  原始图像尺寸
     * @param img       输入图像
     * @param preds     预测结果列表
     * @return 返回检测框、得分和类别名称
     */
    public Triple<List<float[]>, List<Float>, List<String>> call(NDManager manager, Size oriShape, NDArray img, NDList preds) {
        // 分离得分和原始框
        List<NDArray> scores = new ArrayList<>();
        List<NDArray> rawBoxes = new ArrayList<>();
        int numOuts = preds.size() / 2;
        for (int i = 0; i < numOuts; i++) {
            scores.add(preds.get(i));
            rawBoxes.add(preds.get(i + numOuts));
        }

        int batchSize = (int) rawBoxes.get(0).getShape().get(0);
        int regMax = (int) (rawBoxes.get(0).getShape().getLastDimension() / 4 - 1);

        List<Integer> outBoxesNum = new ArrayList<>();
        NDList outBoxesList = new NDList();

        // 获取图像信息
        ImageInfo imageInfo = imgInfo(oriShape, img);
        Shape inputShape = imageInfo.inputShape;
        NDArray scaleFactor = imageInfo.scaleFactor;

        for (int batchId = 0; batchId < batchSize; batchId++) {
            NDList decodeBoxes = new NDList(strides.size());
            NDList selectScores = new NDList(strides.size());

            for (int i = 0; i < strides.size(); i++) {
                int stride = strides.get(i);
                NDArray boxDistribute = rawBoxes.get(i).get(batchId);
                NDArray score = scores.get(i).get(batchId);

                // 生成中心点
                float fmH = (float) inputShape.get(0) / stride;
                float fmW = (float) inputShape.get(1) / stride;

                NDArray hRange = manager.arange(fmH);
                NDArray wRange = manager.arange(fmW);
                float[][] ww = new float[(int)hRange.size()][(int)wRange.size()];
                float[][] hh = new float[(int)hRange.size()][(int)wRange.size()];
                for (int m = 0; m < fmH; m++) {
                    for (int n = 0; n < fmW; n++) {
                        ww[m][n] = wRange.get(n).getFloat();
                        hh[m][n] = hRange.get(m).getFloat();
                    }
                }

                NDArray ctRow = manager.create(hh).flatten().add(0.5).mul(stride);
                NDArray ctCol = manager.create(ww).flatten().add(0.5).mul(stride);
                NDArray center = NDArrays.stack(new NDList(ctCol, ctRow, ctCol, ctRow), 1); // [x, y, x, y]

                // 盒子分布到距离
                NDArray regRange = manager.arange(regMax + 1);
                boxDistribute = boxDistribute.reshape(-1, regMax + 1);
                NDArray boxDistance = softmax(boxDistribute, 1);
                boxDistance = boxDistance.mul(regRange.expandDims(0));
                boxDistance = boxDistance.sum(new int[]{1}).reshape(-1, 4);
                boxDistance = boxDistance.mul(stride);

                // 选择Top K候选
                NDArray topkIdx = score.max(new int[]{1}).argSort().flip(0);
                topkIdx = topkIdx.get(":" + nmsTopK);
                center = center.get(topkIdx);
                score = score.get(topkIdx);
                boxDistance = boxDistance.get(topkIdx);

                // 解码框
                NDArray decodeBox = center.add(boxDistance.mul(manager.create(new float[]{-1, -1, 1, 1})));

                selectScores.add(score);
                decodeBoxes.add(decodeBox);
            }

            // NMS处理
            NDArray bboxes = NDArrays.concat(decodeBoxes);
            NDArray confidences = NDArrays.concat(selectScores);
            NDList pickedBoxProbs = new NDList();
            List<Integer> pickedLabels = new ArrayList<>();

            for (int classIndex = 0; classIndex < confidences.getShape().get(1); classIndex++) {
                NDArray probs = confidences.get(":, {}", classIndex);
                NDArray mask = probs.gt(confThresh);
                probs = probs.get(mask);
                if (mask.sum().getLong() == 0) continue;

                NDArray subsetBoxes = bboxes.get("{}, :", mask);
                NDArray boxProbs = subsetBoxes.concat(probs.reshape(-1, 1), -1);
                boxProbs = hardNms(boxProbs, iouThresh, keepTopK);
                pickedBoxProbs.add(boxProbs);
                for (int i = 0; i < boxProbs.getShape().get(0); i++) {
                    pickedLabels.add(classIndex);
                }
            }

            if (pickedBoxProbs.isEmpty()) {
                // 无检测结果
                outBoxesList.add(manager.create(new float[0][5]));
                outBoxesNum.add(0);
            } else {
                NDArray pickedBoxProbsConcat = NDArrays.concat(pickedBoxProbs);
                // 调整框的大小
                pickedBoxProbsConcat.set(new NDIndex(":, :4"), warpBoxes(manager, pickedBoxProbsConcat.get(":, :4"), oriShape));
                NDArray imScale = NDArrays.concat(new NDList(
                        scaleFactor.get(batchId).flip(0), scaleFactor.get(batchId).flip(0))
                );
                pickedBoxProbsConcat.set(new NDIndex(":, :4"), pickedBoxProbsConcat.get(":, :4").div(imScale));
                // 组合类别、得分和框
                int[] picked_labels = pickedLabels.stream().mapToInt(Integer::intValue).toArray();
                outBoxesList.add(
                        NDArrays.concat(new NDList(
                                manager.create(picked_labels).expandDims(-1),
                                pickedBoxProbsConcat.get(":, 4").expandDims(-1),
                                pickedBoxProbsConcat.get(":, :4")),
                                1)
                );
                outBoxesNum.add(pickedLabels.size());
            }
        }

        // 整合所有批次的结果
        NDArray outBoxesConcat = NDArrays.concat(outBoxesList, 0);
        NDArray outBoxesNumArray = manager.create(outBoxesNum.stream().mapToInt(Integer::intValue).toArray());

        List<float[]> boxes = new ArrayList<>();
        List<Float> scoresList = new ArrayList<>();
        List<String> classNames = new ArrayList<>();

        for (int i = 0; i < outBoxesConcat.getShape().get(0); i++) {
            NDArray dt = outBoxesConcat.get(i);
            int clsId = (int) dt.getFloat(0);
            float[] bbox = dt.get("2:").toFloatArray();
            float score = dt.getFloat(1);
            String label = labels.get(clsId);
            boxes.add(bbox);
            scoresList.add(score);
            classNames.add(label);
        }
        return Triple.of(boxes, scoresList, classNames);
    }

    /**
     * 调整框的大小
     *
     * @param boxes     原始框
     * @param oriShape 原始图像尺寸
     * @return 调整后的框
     */
    private NDArray warpBoxes(NDManager manager, NDArray boxes, Size oriShape) {
        double width =  oriShape.width;
        double height = oriShape.height;
        long n = boxes.getShape().get(0);

        if (n > 0) {
            NDArray xy = manager.ones(new Shape(n * 4, 3));
            xy.set(new NDIndex(":, :2"), boxes.get(":, {}", manager.create(new int[]{0, 1, 2, 3, 0, 3, 2, 1}))
                    .reshape(n * 4, 2));
            // 重新调整框
            xy = (xy.get(":, :2").div(xy.get(":, 2:3"))).reshape(n, 8);
            NDArray x = xy.get(":, {}", manager.create(new int[]{0, 2, 4, 6}));
            NDArray y = xy.get(":, {}", manager.create(new int[]{1, 3, 5, 7}));
            int[] axes1 = new int[]{1};
            xy = NDArrays.concat(new NDList(x.min(axes1), y.min(axes1), x.max(axes1), y.max(axes1))).reshape(4, n).transpose();

            // 裁剪框到图像边界
            xy.set(new NDIndex(":, {}", manager.create(new int[]{0, 2})), xy.get(":, {}", manager.create(new int[]{0, 2})).clip(0, width));
            xy.set(new NDIndex(":, {}", manager.create(new int[]{1, 3})), xy.get(":, {}", manager.create(new int[]{1, 3})).clip(0, height));
            return xy.toType(DataType.FLOAT32, true);
        }
        return boxes;
    }

    /**
     * 获取图像信息，包括原始尺寸、输入尺寸和缩放因子
     *
     * @param originShape 原始图像尺寸
     * @param img         输入图像
     * @return ImageInfo 对象
     */
    private ImageInfo imgInfo(Size originShape, NDArray img) {
        Shape resizeShape = img.getShape();
        float imScaleY = (float) resizeShape.get(2) / (float) originShape.height;
        float imScaleX = (float) resizeShape.get(3) / (float) originShape.width;
        float[] scaleFactor = new float[]{imScaleY, imScaleX};

        Shape imgShape = img.getShape().get(2) > 0 ? new Shape(img.getShape().get(2), img.getShape().get(3)) : new Shape(0, 0);
        return new ImageInfo(originShape, imgShape, img.getManager().create(new float[][]{scaleFactor}));
    }

    /**
     * 计算softmax
     *
     * @param x     输入数组
     * @param axis  轴
     * @return softmax结果
     */
    private NDArray softmax(NDArray x, int axis) {
        NDArray max = x.max(new int[]{axis}, true);
        NDArray stableX = x.sub(max);
        NDArray expX = stableX.exp();
        NDArray sumExp = expX.sum(new int[]{axis}, true);
        return expX.div(sumExp);
    }

    /**
     * 非极大值抑制
     *
     * @param boxScores   框和得分
     * @param iouThresh   IOU阈值
     * @param topK        保留的最大数量
     * @return 经过NMS处理后的框
     */
    private NDArray hardNms(NDArray boxScores, float iouThresh, int topK) {
        NDArray scores = boxScores.get(":, -1");
        NDArray boxes = boxScores.get(":, :-1");
        List<Long> picked = new ArrayList<>();

        // 按分数排序
        NDArray indexes = scores.argSort();

        while (indexes.size(0) > 0) {
            Long current = indexes.getLong(-1);
            picked.add(current);
            if (topK > 0 && topK == picked.size() || indexes.size() == 1) {
                break;
            }
            NDArray currentBox = boxes.get("{}, :", current);
            indexes = indexes.get(":-1");
            NDArray restBoxes = boxes.get("{}, :", indexes);
            NDArray iou = iouOf(restBoxes, currentBox);
            NDArray mask = iou.lte(iouThresh);
            indexes = indexes.get(mask);
        }

        // 返回选中的框
        NDArray pickedArray = boxScores.getManager().create(picked.stream().mapToInt(Long::intValue).toArray());
        return boxScores.get("{}, :", pickedArray);
    }

    /**
     * 计算两个框的IoU
     *
     * @param boxes0 第一个框
     * @param boxes1 第二个框
     * @return IoU值
     */
    private NDArray iouOf(NDArray boxes0, NDArray boxes1) {
        NDArray overlapLeftTop = NDArrays.maximum(boxes0.get("..., :2"), boxes1.get("..., :2"));
        NDArray overlapRightBottom = NDArrays.maximum(boxes0.get("..., 2:"), boxes1.get("..., 2:"));

        NDArray overlapArea = areaOf(overlapLeftTop, overlapRightBottom);
        NDArray area0 = areaOf(boxes0.get("..., :2"), boxes0.get("..., 2:"));
        NDArray area1 = areaOf(boxes1.get("..., :2"), boxes1.get("..., 2:"));

        return overlapArea.div(area0.add(area1).sub(overlapArea).add(1e-5f));
    }

    /**
     * 计算面积
     *
     * @param leftTop       左上角
     * @param rightBottom   右下角
     * @return 面积
     */
    private NDArray areaOf(NDArray leftTop, NDArray rightBottom) {
        NDArray hw = rightBottom.sub(leftTop).clip(0, Float.MAX_VALUE); // 修正这里
        return hw.get("..., 0").mul(hw.get("..., 1"));
    }

    /**
     * ImageInfo 类用于存储图像信息
     */
    private static class ImageInfo {
        Size oriShape;
        Shape inputShape;
        NDArray scaleFactor;

        ImageInfo(Size oriShape, Shape inputShape, NDArray scaleFactor) {
            this.oriShape = oriShape;
            this.inputShape = inputShape;
            this.scaleFactor = scaleFactor;
        }
    }
}
