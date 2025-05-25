package io.github.hzkitty.rapidlayout;

import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OrtException;
import io.github.hzkitty.rapidlayout.entity.*;
import io.github.hzkitty.rapidlayout.utils.DownloadModel;
import io.github.hzkitty.rapidlayout.utils.LoadImage;
import io.github.hzkitty.rapidlayout.utils.NDArrayUtils;
import io.github.hzkitty.rapidlayout.utils.OrtInferSession;
import io.github.hzkitty.rapidlayout.utils.post.DocLayoutPostProcess;
import io.github.hzkitty.rapidlayout.utils.post.PPPostProcess;
import io.github.hzkitty.rapidlayout.utils.post.YOLOv8PostProcess;
import io.github.hzkitty.rapidlayout.utils.pre.DocLayoutPreProcess;
import io.github.hzkitty.rapidlayout.utils.pre.PPPreProcess;
import io.github.hzkitty.rapidlayout.utils.pre.YOLOv8PreProcess;
import io.github.hzkitty.rapidlayout.utils.OpencvLoader;
import org.opencv.core.Mat;
import org.opencv.core.Size;

import java.awt.image.BufferedImage;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

import static io.github.hzkitty.rapidlayout.entity.LayoutModelType.*;

public class RapidLayout {

    private static final Logger logger = Logger.getLogger(RapidLayout.class.getName());

//    // 远程模型下载地址前缀
    private static final String ROOT_URL = "https://github.com/RapidAI/RapidLayout/releases/download/v0.0.0/";

    // 关键字 -> 模型下载地址 的映射
    private static final Map<LayoutModelType, String> KEY_TO_MODEL_URL = new HashMap<LayoutModelType, String>() {{
        put(PP_LAYOUT_TABLE,       ROOT_URL + "layout_cdla.onnx");
        put(PP_LAYOUT_PUBLAYNET,  ROOT_URL + "layout_publaynet.onnx");
        put(PP_LAYOUT_CDLA,      ROOT_URL + "layout_table.onnx");
        put(YOLOV8N_LAYOUT_PAPER, ROOT_URL + "yolov8n_layout_paper.onnx");
        put(YOLOV8N_LAYOUT_REPORT, ROOT_URL + "yolov8n_layout_report.onnx");
        put(YOLOV8N_LAYOUT_PUBLAYNET, ROOT_URL + "yolov8n_layout_publaynet.onnx");
        put(YOLOV8N_LAYOUT_GENERAL6, ROOT_URL + "yolov8n_layout_general6.onnx");
        put(DOCLAYOUT_DOCSTRUCTBENCH,       ROOT_URL + "doclayout_yolo_docstructbench_imgsz1024.onnx");
        put(DOCLAYOUT_D4LA,       ROOT_URL + "doclayout_yolo_d4la_imgsz1600_docsynth_pretrain.onnx");
        put(DOCLAYOUT_DOCSYNTH,       ROOT_URL + "doclayout_yolo_d4la_imgsz1600_docsynth_pretrain.onnx");
    }};

    // 默认模型路径
//    private static final String DEFAULT_MODEL_PATH = Paths.get("models", "layout_cdla.onnx").toString();

    // 模型类型
    private final LayoutModelType modelType;
    // 推理会话 (ONNX Runtime)
    private final OrtInferSession session;

    // --- 前后处理工具 ---
    private final PPPreProcess ppPreProcess;
    private final PPPostProcess ppPostProcess;

    private final YOLOv8PreProcess yoloPreProcess;
    private final YOLOv8PostProcess yoloPostProcess;

    private final DocLayoutPreProcess doclayoutPreProcess;
    private final DocLayoutPostProcess doclayoutPostProcess;

    // 记录各自的输入大小
    private final int[] yoloInputShape = new int[]{640, 640};
    private final int[] doclayoutShape = new int[]{1024, 1024};

    // 图片加载器
    private final LoadImage loadImg;

    // 用于区分三种模型类型的列表
    private final List<LayoutModelType> ppLayoutType;
    private final List<LayoutModelType> yoloLayoutType;
    private final List<LayoutModelType> docLayoutType;

    public static RapidLayout create() {
        return new RapidLayout();
    }

    public static RapidLayout create(LayoutConfig config) {
        return new RapidLayout(config);
    }

    public RapidLayout() {
        this(new LayoutConfig());
    }

    public RapidLayout(LayoutConfig config) {
        OpencvLoader.loadOpencvLib();
        // 校验阈值
        if (!checkOf(config.confThres)) {
            throw new IllegalArgumentException("conf_thres " + config.confThres + " 超出 [0,1] 范围");
        }
        if (!checkOf(config.iouThres)) {
            throw new IllegalArgumentException("iou_thres " + config.iouThres + " 超出 [0,1] 范围");
        }
        this.modelType = config.modelType;

        // 确定最终模型路径 (本地 or 下载)
//        String finalModelPath = getModelPath(modelType, config.modelPath);

        // 构建 session 配置
        OrtInferConfig inferConfig = new OrtInferConfig();
        inferConfig.setModelPath(config.modelPath);
        inferConfig.setUseCuda(config.useCuda);
        inferConfig.setDeviceId(config.deviceId);

        // 初始化 ONNXRuntime session
        this.session = new OrtInferSession(inferConfig);
        List<String> labels = this.session.getCharacterList("character");
        logger.info(modelType + " contains " + labels);

        // 初始化三种前处理 & 后处理
        this.ppPreProcess = new PPPreProcess(new Size(608, 800));
        this.ppPostProcess = new PPPostProcess(labels, config.confThres, config.iouThres);

        this.yoloPreProcess = new YOLOv8PreProcess(yoloInputShape[0], yoloInputShape[1]);
        this.yoloPostProcess = new YOLOv8PostProcess(labels, config.confThres, config.iouThres);

        this.doclayoutPreProcess  = new DocLayoutPreProcess(doclayoutShape[0], doclayoutShape[1]);
        this.doclayoutPostProcess = new DocLayoutPostProcess(labels, config.confThres, config.iouThres);

        // 加载图片的工具
        this.loadImg = new LoadImage();

        // 分组三种模型类型
        this.ppLayoutType  = new ArrayList<>();
        this.yoloLayoutType= new ArrayList<>();
        this.docLayoutType = new ArrayList<>();
        // 根据 KEY_TO_MODEL_URL 里的 key 判断归属
        for (LayoutModelType k : KEY_TO_MODEL_URL.keySet()) {
            if (k.getModelName().startsWith("pp_")) {
                ppLayoutType.add(k);
            } else if (k.getModelName().startsWith("yolov8n_")) {
                yoloLayoutType.add(k);
            } else if (k.getModelName().startsWith("doclayout")) {
                docLayoutType.add(k);
            }
        }
    }

    public LayoutResult run(String imagePath) throws Exception {
        return this.runImpl(imagePath);
    }

    public LayoutResult run(Path imagePath) throws Exception {
        return this.runImpl(imagePath);
    }

    public LayoutResult run(byte[] imageData) throws Exception {
        return this.runImpl(imageData);
    }

    public LayoutResult run(BufferedImage image) throws Exception {
        return this.runImpl(image);
    }

    public LayoutResult run(Mat mat) throws Exception {
        return this.runImpl(mat);
    }

    /**
     * 供外部调用的推理接口
     * @param imgContent  图片输入(路径/字节/矩阵)
     * @return LayoutResult: { boxes, scores, classNames, elapsed }
     */
    private LayoutResult runImpl(Object imgContent) throws Exception {
        // 1. 加载图片
        Mat img = this.loadImg.call(imgContent);
        Size oriImgShape = img.size();

        // 2. 判断模型类型并调用对应逻辑
        if (ppLayoutType.contains(modelType)) {
            try (NDManager manager = NDManager.newBaseManager()) {
                return ppLayout(manager, img, oriImgShape);
            }
        }
        if (yoloLayoutType.contains(modelType)) {
            return yolov8Layout(img, oriImgShape);
        }
        if (docLayoutType.contains(modelType)) {
            return doclayoutLayout(img, oriImgShape);
        }
        throw new IllegalArgumentException(modelType + " 不受支持");
    }

    private LayoutResult ppLayout(NDManager manager, Mat img, Size oriImgShape) throws OrtException {
        long startTime = System.currentTimeMillis();

        // 1) 前处理 => 返回四维数组(如 float[1][3][H][W])
        float[][][][] inputData = ppPreProcess.call(img);

        // 2) session 推理 => 返回网络 preds
        Object[] outputs = session.run(inputData);
        // 确保 preds 数组的大小与 outputs 相同
//        float[][][][] preds = new float[outputs.length][][][];
        NDList preds = new NDList(outputs.length);
        // 遍历 outputs 并赋值给 preds
        for (Object output : outputs) {
            preds.add(NDArrayUtils.create(manager, (float[][][]) output));
        }
        // 3) 后处理 => (boxes, scores, classNames)
        Triple<List<float[]>, List<Float>, List<String>> result = ppPostProcess.call(manager, oriImgShape, NDArrayUtils.create(manager, inputData), preds);
        double elapse = (System.currentTimeMillis() - startTime) / 1000.0;
        return new LayoutResult(result.getLeft(), result.getMiddle(), result.getRight(), elapse);

    }

    private LayoutResult yolov8Layout(Mat img, Size oriImgShape) throws OrtException {
        long startTime = System.currentTimeMillis();
        // 前处理
        float[][][][] inputTensor = yoloPreProcess.call(img);
        // 推理
        Object[] outputs = session.run(inputTensor);
        float[][][][] preds = new float[outputs.length][][][];
        for (int i = 0; i < outputs.length; i++) {
            preds[i] = (float[][][]) outputs[i];
        }
        // 3) 后处理 => (boxes, scores, classNames)
        Triple<List<float[]>, List<Float>, List<String>> result = yoloPostProcess.call(preds, oriImgShape, this.yoloInputShape);
        double elapse = (System.currentTimeMillis() - startTime) / 1000.0;
        return new LayoutResult(result.getLeft(), result.getMiddle(), result.getRight(), elapse);
    }

    private LayoutResult doclayoutLayout(Mat img, Size oriImgShape) throws OrtException {
        long startTime = System.currentTimeMillis();
        // 前处理
        float[][][][] inputTensor = doclayoutPreProcess.call(img);
        // 推理
        Object[] outputs = session.run(inputTensor);
        float[][][][] preds = new float[outputs.length][][][];
        for (int i = 0; i < outputs.length; i++) {
            preds[i] = (float[][][]) outputs[i];
        }
        // 3) 后处理 => (boxes, scores, classNames)
        Triple<List<float[]>, List<Float>, List<String>> result = doclayoutPostProcess.call(preds, oriImgShape, this.doclayoutShape);
        double elapse = (System.currentTimeMillis() - startTime) / 1000.0;
        return new LayoutResult(result.getLeft(), result.getMiddle(), result.getRight(), elapse);
    }

//    /**
//     * 传入 modelType/modelPath, 如果 modelPath 不为空则直接用它，
//     * 否则根据 KEY_TO_MODEL_URL 下载对应的 onnx，再返回本地路径
//     */
//    private static String getModelPath(LayoutModelType modelType, String modelPath) throws DownloadModel.DownloadModelError {
//        if (modelPath != null && !modelPath.isEmpty()) {
//            return modelPath;
//        }
//        String url = KEY_TO_MODEL_URL.getOrDefault(modelType, null);
//        if (url != null) {
//            String localPath = DownloadModel.download(url);
//            return localPath;
//        }
//        logger.info("model url is null, 使用默认模型 " + DEFAULT_MODEL_PATH);
//        return DEFAULT_MODEL_PATH;
//    }

    /**
     * 检查阈值取值合法性: [0,1]
     */
    private static boolean checkOf(float thres) {
        return thres >= 0.0f && thres <= 1.0f;
    }

}
