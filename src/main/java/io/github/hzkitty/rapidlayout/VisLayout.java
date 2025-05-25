package io.github.hzkitty.rapidlayout;

import org.opencv.core.*;
import org.opencv.core.Core;
import org.opencv.imgproc.Imgproc;

import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

/**
 * VisLayout - 在图像上绘制检测结果（边框、标签、分数以及遮罩）
 */
public class VisLayout {

    /**
     * 在图像上绘制检测框、分数和类别名称，并且绘制遮罩
     *
     * @param image      原始图像（Mat类型，H x W x C）
     * @param boxes      检测框列表，每个元素为长度为 4 的 float[]，对应 [x1, y1, x2, y2]
     * @param scores     每个检测框对应的置信度分数
     * @param classNames 每个检测框对应的类别名称
     * @param maskAlpha  遮罩透明度，取值范围通常在 [0,1]，值越大遮罩越深
     * @return           绘制好结果的图像（Mat）
     */
    public static Mat drawDetections(Mat image, List<float[]> boxes, List<Float> scores, List<String> classNames, float maskAlpha) {
        if (boxes == null || scores == null || classNames == null) {
            return null;
        }

        // 复制原图，以防修改原图像
        Mat detImg = image.clone();

        // 取得图像的高和宽
        int imgHeight = detImg.rows();
        int imgWidth = detImg.cols();

        // 根据图像大小设置字体大小和字体粗细
        double fontSize = Math.min(imgHeight, imgWidth) * 0.0006;
        int textThickness = (int) (Math.min(imgHeight, imgWidth) * 0.001);

        // 先绘制所有检测框的遮罩
        detImg = drawMasks(detImg, boxes, maskAlpha);

        // 遍历每个检测框，绘制边框与文本
        for (int i = 0; i < boxes.size(); i++) {
            float[] box = boxes.get(i);
            float score = scores.get(i);
            String label = classNames.get(i);

            // 随机获取一个颜色
            Scalar color = getColor();

            // 绘制矩形框
            drawBox(detImg, box, color, 2);

            // 准备文本 (类别 + 分数)
            String caption = String.format("%s %d%%", label, (int) (score * 100));

            // 绘制文本
            drawText(detImg, caption, box, color, fontSize, textThickness);
        }

        return detImg;
    }

    /**
     * 绘制矩形框
     *
     * @param image     Mat 类型的图像
     * @param box       float[]，长度为 4 的数组，对应 [x1, y1, x2, y2]
     * @param color     矩形框颜色 (B, G, R)
     * @param thickness 矩形框线条粗细
     */
    private static void drawBox(Mat image, float[] box, Scalar color, int thickness) {
        int x1 = (int) box[0];
        int y1 = (int) box[1];
        int x2 = (int) box[2];
        int y2 = (int) box[3];

        // 在图像上绘制矩形，注意坐标点用 Point 表示
        Imgproc.rectangle(image, new Point(x1, y1), new Point(x2, y2), color, thickness);
    }

    /**
     * 绘制文本（标签与分数）
     *
     * @param image         Mat 类型图像
     * @param text          要绘制的文本（类别 + 分数）
     * @param box           float[]，长度为 4 的数组，对应 [x1, y1, x2, y2]（可用于定位文本）
     * @param color         文本背景色 (B, G, R)
     * @param fontSize      字体大小
     * @param textThickness 文本线条粗细
     */
    private static void drawText(Mat image, String text, float[] box, Scalar color, double fontSize, int textThickness) {
        int x1 = (int) box[0];
        int y1 = (int) box[1];
        // int x2 = (int) box[2];
        // int y2 = (int) box[3];

        // 计算文本尺寸
        Size textSize = Imgproc.getTextSize(
                text,
                Imgproc.FONT_HERSHEY_SIMPLEX,
                fontSize,
                textThickness,
                null
        );
        int tw = (int) textSize.width;
        int th = (int) textSize.height;

        // 在文本区域画一个实心矩形作为背景（为了让文字清晰）
        // 这里将矩形高度稍微加大一点
        Imgproc.rectangle(
                image,
                new Point(x1, y1),
                new Point(x1 + tw, y1 - (int) (th * 1.2)),
                color,
                -1
        );

        // 将文本叠加在实心矩形上方
        Imgproc.putText(
                image,
                text,
                new Point(x1, y1),
                Imgproc.FONT_HERSHEY_SIMPLEX,
                fontSize,
                new Scalar(255, 255, 255), // 白色字体
                textThickness
        );
    }

    /**
     * 对所有检测框进行遮罩绘制，然后与原图进行加权融合
     *
     * @param image     Mat 类型图像
     * @param boxes     包含所有检测框的列表
     * @param maskAlpha 遮罩透明度
     * @return          已融合遮罩的图像
     */
    private static Mat drawMasks(Mat image, List<float[]> boxes, float maskAlpha) {
        // 克隆一份图像，用于绘制矩形遮罩
        Mat maskImg = image.clone();

        // 遍历每个检测框并画出实心矩形
        for (float[] box : boxes) {
            Scalar color = getColor();
            int x1 = (int) box[0];
            int y1 = (int) box[1];
            int x2 = (int) box[2];
            int y2 = (int) box[3];

            // 绘制实心矩形
            Imgproc.rectangle(
                    maskImg,
                    new Point(x1, y1),
                    new Point(x2, y2),
                    color,
                    -1
            );
        }

        // 使用 addWeighted 实现图像融合（maskAlpha 越大，遮罩越明显）
        Core.addWeighted(
                maskImg,          // 前景图像
                maskAlpha,        // 前景权重
                image,            // 背景图像
                1.0 - maskAlpha,  // 背景权重
                0.0,              // 亮度调整量
                image             // 融合到原图
        );

        return image;
    }

    /**
     * 生成随机颜色
     *
     * @return 随机颜色 (B, G, R)
     */
    private static Scalar getColor() {
        Random random = new Random();
        double b = random.nextDouble() * 255;
        double g = random.nextDouble() * 255;
        double r = random.nextDouble() * 255;
        return new Scalar(b, g, r);
    }


    private List<Mat> getCropImgList(Mat img, List<Point[]> dtBoxes) {
        return (List)dtBoxes.stream().map((box) -> {
            return this.getRotateCropImage(img, box);
        }).collect(Collectors.toList());
    }

    private Mat getRotateCropImage(Mat img, Point[] points) {
        double widthTop = this.distance(points[0], points[1]);
        double widthBottom = this.distance(points[2], points[3]);
        int imgCropWidth = (int)Math.max(widthTop, widthBottom);
        double heightLeft = this.distance(points[0], points[3]);
        double heightRight = this.distance(points[1], points[2]);
        int imgCropHeight = (int)Math.max(heightLeft, heightRight);
        MatOfPoint2f ptsStd = new MatOfPoint2f(new Point[]{new Point(0.0, 0.0), new Point((double)imgCropWidth, 0.0), new Point((double)imgCropWidth, (double)imgCropHeight), new Point(0.0, (double)imgCropHeight)});
        MatOfPoint2f ptsSrc = new MatOfPoint2f(points);
        Mat perspectiveTransform = Imgproc.getPerspectiveTransform(ptsSrc, ptsStd);
        Mat dstImg = new Mat();
        Imgproc.warpPerspective(img, dstImg, perspectiveTransform, new Size((double)imgCropWidth, (double)imgCropHeight), 2, 1);
        if ((double)dstImg.rows() / (double)dstImg.cols() >= 1.5) {
            Core.rotate(dstImg, dstImg, 0);
        }

        return dstImg;
    }

    private double distance(Point p1, Point p2) {
        return Math.hypot(p1.x - p2.x, p1.y - p2.y);
    }

    /**
     * 根据传入的 bbox (x1, y1, x2, y2) 裁剪出子图。
     *
     * @param image 原始图像（Mat），在 OpenCV 中通常为 BGR 格式。
     * @param box   float[]，长度为 4，对应 [x1, y1, x2, y2]。
     * @return      裁剪后的图像 (Mat)。
     */
    public static Mat cropImage(Mat image, float[] box) {
        // 获取图像宽高
        int width = image.cols();
        int height = image.rows();

        // 解析 bbox，注意进行 int 转换
        int x1 = Math.round(box[0]);
        int y1 = Math.round(box[1]);
        int x2 = Math.round(box[2]);
        int y2 = Math.round(box[3]);

        // 如果有需要，可做 x1, y1, x2, y2 的顺序校正
        // 比如确保 x1 < x2, y1 < y2
        if (x1 > x2) {
            int tmp = x1;
            x1 = x2;
            x2 = tmp;
        }
        if (y1 > y2) {
            int tmp = y1;
            y1 = y2;
            y2 = tmp;
        }

        // 防止坐标越界，可以对坐标进行 clamp 操作
        // 确保 x1, y1 不小于 0, 且 x2, y2 不大于图像边界
        x1 = Math.max(0, Math.min(x1, width - 1));
        y1 = Math.max(0, Math.min(y1, height - 1));
        x2 = Math.max(0, Math.min(x2, width - 1));
        y2 = Math.max(0, Math.min(y2, height - 1));

        // 计算矩形区域的宽高
        int cropWidth = x2 - x1;
        int cropHeight = y2 - y1;

        // 如果裁剪区域无效（宽或高 <= 0），返回空 Mat 或者自行处理
        if (cropWidth <= 0 || cropHeight <= 0) {
            return new Mat();
        }

        // 使用 OpenCV 的 Rect 和 Mat 构造裁剪区域
        Rect roi = new Rect(x1, y1, cropWidth, cropHeight);

        // 从原图中截取该区域；clone() 以防与原图共享数据导致后续修改冲突
        return new Mat(image, roi).clone();
    }
}
