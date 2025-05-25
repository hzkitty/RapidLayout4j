package io.github.hzkitty.rapidlayout.entity;

import java.util.Arrays;
import java.util.List;

/**
 * 版面识别结果
 */
public class LayoutResult {
    public List<float[]> boxes;
    public List<Float> scores;
    public List<String> classNames;
    public double elapse;

    public LayoutResult(List<float[]> boxes, List<Float> scores, List<String> classNames, double elapse) {
        this.boxes = boxes;
        this.scores = scores;
        this.classNames = classNames;
        this.elapse = elapse;
    }

    public List<float[]> getBoxes() {
        return boxes;
    }

    public void setBoxes(List<float[]> boxes) {
        this.boxes = boxes;
    }

    public List<Float> getScores() {
        return scores;
    }

    public void setScores(List<Float> scores) {
        this.scores = scores;
    }

    public List<String> getClassNames() {
        return classNames;
    }

    public void setClassNames(List<String> classNames) {
        this.classNames = classNames;
    }

    public double getElapse() {
        return elapse;
    }

    public void setElapse(double elapse) {
        this.elapse = elapse;
    }

    @Override
    public String toString() {
        return "LayoutResult{" +
                "boxes=" + (boxes != null ? Arrays.deepToString(boxes.toArray()) : "null") +
                ", scores=" + scores +
                ", classNames=" + classNames +
                ", elapse=" + elapse +
                '}';
    }
}