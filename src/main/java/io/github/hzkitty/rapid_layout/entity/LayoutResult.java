package io.github.hzkitty.rapid_layout.entity;

import java.util.List;

/**
 * 版面识别结果
 */
public class LayoutResult {
    public List<float[]> boxes;
    public List<Float> scores;
    public List<String> classNames;
    public double elapsed;

    public LayoutResult(List<float[]> boxes, List<Float> scores, List<String> classNames, double elapsed) {
        this.boxes = boxes;
        this.scores = scores;
        this.classNames = classNames;
        this.elapsed = elapsed;
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

    public double getElapsed() {
        return elapsed;
    }

    public void setElapsed(double elapsed) {
        this.elapsed = elapsed;
    }
}