
import io.github.hzkitty.rapidlayout.RapidLayout;
import io.github.hzkitty.rapidlayout.entity.LayoutResult;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.file.Files;

public class LayoutTest {

//    static {
//        nu.pattern.OpenCV.loadShared();
//    }

    @Test
    public void testPath() throws Exception {
        RapidLayout rapidLayout = RapidLayout.create();
        File file = new File("src/test/resources/layout.png");
        String imgContent = file.getAbsolutePath();
        LayoutResult layoutResult = rapidLayout.run(imgContent);
        Assertions.assertFalse(layoutResult.getBoxes().isEmpty());
        System.out.println(layoutResult);
    }

    @Test
    public void testBufferedImage() throws Exception {
        RapidLayout rapidLayout = RapidLayout.create();
        File file = new File("src/test/resources/layout.png");
        BufferedImage imgContent = ImageIO.read(file);
        LayoutResult layoutResult = rapidLayout.run(imgContent);
        Assertions.assertFalse(layoutResult.getBoxes().isEmpty());
        System.out.println(layoutResult);
    }

    @Test
    public void testByte() throws Exception {
        RapidLayout rapidLayout = RapidLayout.create();
        File file = new File("src/test/resources/layout.png");
        byte[] imgContent = Files.readAllBytes(file.toPath());
        LayoutResult layoutResult = rapidLayout.run(imgContent);
        Assertions.assertFalse(layoutResult.getBoxes().isEmpty());
        System.out.println(layoutResult);
    }

    @Test
    public void testMat() throws Exception {
        RapidLayout rapidLayout = RapidLayout.create();
        File file = new File("src/test/resources/layout.png");
        Mat imgContent = Imgcodecs.imread(file.getAbsolutePath());
        LayoutResult layoutResult = rapidLayout.run(imgContent);
        Assertions.assertFalse(layoutResult.getBoxes().isEmpty());
        System.out.println(layoutResult);
    }

}
