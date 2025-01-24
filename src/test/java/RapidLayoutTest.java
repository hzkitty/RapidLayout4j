import io.github.hzkitty.rapid_layout.RapidLayout;
import io.github.hzkitty.rapid_layout.VisLayout;
import io.github.hzkitty.rapid_layout.entity.LayoutResult;
import io.github.hzkitty.rapid_layout.utils.LoadImage;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;

public class RapidLayoutTest {

    @Test
    public void testLayout() throws Exception {
//        String modelType = "pp_layout_cdla";
//        File modelFile = new File("src/test/resources/models/layout_cdla.onnx");

        String modelType = "doclayout_yolo";
        File modelFile = new File("src/test/resources/models/doclayout_yolo_docstructbench_imgsz1024.onnx");
        String modelPath = modelFile.getAbsolutePath();
        RapidLayout layout = new RapidLayout(modelType, modelPath, 0.2f, 0.5f, false);

        // 调用
        File file = new File("src/test/resources/pdf_02.jpg");
        String imgContent = file.getAbsolutePath();
        LayoutResult result = layout.call(imgContent);
        Assertions.assertFalse(result.getBoxes().isEmpty());

        System.out.println("检测到: " + result.boxes.size() + " 个框");
        for (int i = 0; i < result.boxes.size(); i++) {
            System.out.printf("box: %s, score=%.2f, class=%s%n",
                    Arrays.toString(result.boxes.get(i)),
                    result.scores.get(i),
                    result.classNames.get(i));
        }
        System.out.println("推理耗时: " + result.elapsed + "秒");

        LoadImage loadImg = new LoadImage();
        Mat img = loadImg.call(imgContent);
        Mat plotedImg = VisLayout.drawDetections(img, result.boxes, result.scores, result.classNames, 0.3f);

        Path saveDir = Paths.get("src/test/resources/inference_results").toAbsolutePath();
        Files.createDirectories(saveDir);
        Path saveLayoutPath = saveDir.resolve("layout_res.png");
        Imgcodecs.imwrite(saveLayoutPath.toString(), plotedImg);

    }
}
