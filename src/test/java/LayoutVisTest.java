import io.github.hzkitty.rapidlayout.RapidLayout;
import io.github.hzkitty.rapidlayout.VisLayout;
import io.github.hzkitty.rapidlayout.entity.LayoutConfig;
import io.github.hzkitty.rapidlayout.entity.LayoutModelType;
import io.github.hzkitty.rapidlayout.entity.LayoutResult;
import io.github.hzkitty.rapidlayout.utils.LoadImage;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;

public class LayoutVisTest {

    @Test
    public void testLayout() throws Exception {
        File modelFile = new File("src/test/resources/models/doclayout_yolo_docstructbench_imgsz1024.onnx");
        LayoutConfig config = new LayoutConfig();
        config.setModelType(LayoutModelType.DOCLAYOUT_DOCSTRUCTBENCH);
        config.setModelPath(modelFile.getAbsolutePath());
        RapidLayout layout = RapidLayout.create(config);

        // 调用
        File file = new File("src/test/resources/layout.png");
        String imgContent = file.getAbsolutePath();
        LayoutResult result = layout.run(imgContent);
        Assertions.assertFalse(result.getBoxes().isEmpty());

        System.out.println("检测到: " + result.boxes.size() + " 个框");
        for (int i = 0; i < result.boxes.size(); i++) {
            System.out.printf("box: %s, score=%.2f, class=%s%n",
                    Arrays.toString(result.boxes.get(i)),
                    result.scores.get(i),
                    result.classNames.get(i));
        }
        System.out.println("推理耗时: " + result.elapse + "秒");

        LoadImage loadImg = new LoadImage();
        Mat img = loadImg.call(imgContent);
        Mat plotedImg = VisLayout.drawDetections(img, result.boxes, result.scores, result.classNames, 0.3f);

        Path saveDir = Paths.get("src/test/resources/inference_results").toAbsolutePath();
        Files.createDirectories(saveDir);
        Path saveLayoutPath = saveDir.resolve("layout_res.png");
        Imgcodecs.imwrite(saveLayoutPath.toString(), plotedImg);

    }
}
