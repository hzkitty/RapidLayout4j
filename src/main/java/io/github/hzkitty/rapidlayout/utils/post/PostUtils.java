package io.github.hzkitty.rapidlayout.utils.post;

import java.util.*;
import java.util.stream.IntStream;

public class PostUtils {

    /**
     * 将 [x,y,w,h] 从 inputShape 缩放回原图尺寸
     */
    public static float[][] rescaleBoxes(float[][] boxes, int inputW, int inputH, int imgW, int imgH) {
        float[][] out = new float[boxes.length][4];
        for (int i = 0; i < boxes.length; i++) {
            // boxes[i]: [x, y, w, h] in model input size
            float x = boxes[i][0] / inputW * imgW;
            float y = boxes[i][1] / inputH * imgH;
            float w = boxes[i][2] / inputW * imgW;
            float h = boxes[i][3] / inputH * imgH;
            out[i][0] = x;
            out[i][1] = y;
            out[i][2] = w;
            out[i][3] = h;
        }
        return out;
    }

    /**
     * 将 xywh => xyxy
     */
    public static float[][] xywh2xyxy(float[][] boxes) {
        float[][] out = new float[boxes.length][4];
        for (int i = 0; i < boxes.length; i++) {
            float x = boxes[i][0];
            float y = boxes[i][1];
            float w = boxes[i][2];
            float h = boxes[i][3];
            // x1 = x - w/2
            // y1 = y - h/2
            // x2 = x + w/2
            // y2 = y + h/2
            float x1 = (float) (x - w / 2.0);
            float y1 = (float) (y - h / 2.0);
            float x2 = (float) (x + w / 2.0);
            float y2 = (float) (y + h / 2.0);
            out[i][0] = x1;
            out[i][1] = y1;
            out[i][2] = x2;
            out[i][3] = y2;
        }
        return out;
    }

    /**
     * 多类别 NMS
     */
    public static int[] multiclassNms(float[][] boxes, float[] scores, int[] classIds, double iouThreshold) {
        // 将相同 classId 的元素分组，然后分别执行 nms，最后拼回
        Map<Integer, List<Integer>> class2indices = new HashMap<>();
        for (int i = 0; i < classIds.length; i++) {
            class2indices.computeIfAbsent(classIds[i], k -> new ArrayList<>()).add(i);
        }

        List<Integer> keep = new ArrayList<>();
        for (Map.Entry<Integer, List<Integer>> e : class2indices.entrySet()) {
            int clsId = e.getKey();
            List<Integer> idxList = e.getValue();

            // 准备 classBoxes, classScores
            float[][] classBoxes = new float[idxList.size()][4];
            float[] classScores = new float[idxList.size()];
            for (int i = 0; i < idxList.size(); i++) {
                int idx = idxList.get(i);
                classBoxes[i] = boxes[idx];
                classScores[i] = scores[idx];
            }
        }
        // 转 int[]
        return keep.stream().mapToInt(i -> i).toArray();
    }

    /**
     * 标准 NMS (单类)
     */
    public static List<Integer> nms(float[][] boxes, float[] scores, float iouThreshold) {
        // 按置信度降序排序
        Integer[] indices = IntStream.range(0, scores.length).boxed()
                .sorted((i, j) -> Double.compare(scores[j], scores[i]))
                .toArray(Integer[]::new);

        List<Integer> keep = new ArrayList<>();
        List<Integer> sortedList = new ArrayList<>(Arrays.asList(indices));

        while (!sortedList.isEmpty()) {
            int current = sortedList.remove(0);
            keep.add(current);

            float[] currentBox = boxes[current];
            // 计算 IoU，过滤过高的
            List<Integer> remain = new ArrayList<>();
            for (int idx : sortedList) {
                double iouVal = computeIou(currentBox, boxes[idx]);
                if (iouVal <= iouThreshold) {
                    remain.add(idx);
                }
            }
            sortedList = remain;
        }
        return keep;
    }

    /**
     * 计算 IoU
     */
    public static float computeIou(float[] boxA, float[] boxB) {
        float interLeft   = Math.max(boxA[0], boxB[0]);
        float interTop    = Math.max(boxA[1], boxB[1]);
        float interRight  = Math.min(boxA[2], boxB[2]);
        float interBottom = Math.min(boxA[3], boxB[3]);

        float interWidth  = Math.max(0, interRight - interLeft);
        float interHeight = Math.max(0, interBottom - interTop);
        float interArea   = interWidth * interHeight;

        float areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]);
        float areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]);
        return (float) (interArea / (areaA + areaB - interArea + 1e-5));
    }

}
