package io.github.hzkitty.rapid_layout;

import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OrtException;
import io.github.hzkitty.rapid_layout.entity.LayoutResult;
import io.github.hzkitty.rapid_layout.entity.OrtInferConfig;
import io.github.hzkitty.rapid_layout.entity.Triple;
import io.github.hzkitty.rapid_layout.utils.DownloadModel;
import io.github.hzkitty.rapid_layout.utils.LoadImage;
import io.github.hzkitty.rapid_layout.utils.NDArrayUtils;
import io.github.hzkitty.rapid_layout.utils.OrtInferSession;
import io.github.hzkitty.rapid_layout.utils.post.DocLayoutPostProcess;
import io.github.hzkitty.rapid_layout.utils.post.PPPostProcess;
import io.github.hzkitty.rapid_layout.utils.post.YOLOv8PostProcess;
import io.github.hzkitty.rapid_layout.utils.pre.DocLayoutPreProcess;
import io.github.hzkitty.rapid_layout.utils.pre.PPPreProcess;
import io.github.hzkitty.rapid_layout.utils.pre.YOLOv8PreProcess;
import io.github.hzkitty.rapid_layout.utils.OpencvLoader;
import org.opencv.core.Mat;
import org.opencv.core.Size;

import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

public class RapidLayout {

    private static final Logger logger = Logger.getLogger(RapidLayout.class.getName());

    // 远程模型下载地址前缀
    private static final String ROOT_URL = "https://github.com/RapidAI/RapidLayout/releases/download/v0.0.0/";

    // 关键字 -> 模型下载地址 的映射
    private static final Map<String, String> KEY_TO_MODEL_URL = new HashMap<String, String>() {{
        put("pp_layout_cdla",       ROOT_URL + "layout_cdla.onnx");
        put("pp_layout_publaynet",  ROOT_URL + "layout_publaynet.onnx");
        put("pp_layout_table",      ROOT_URL + "layout_table.onnx");
        put("yolov8n_layout_paper", ROOT_URL + "yolov8n_layout_paper.onnx");
        put("yolov8n_layout_report",ROOT_URL + "yolov8n_layout_report.onnx");
        put("yolov8n_layout_publaynet", ROOT_URL + "yolov8n_layout_publaynet.onnx");
        put("yolov8n_layout_general6", ROOT_URL + "yolov8n_layout_general6.onnx");
        put("doclayout_yolo",       ROOT_URL + "doclayout_yolo_docstructbench_imgsz1024.onnx");
    }};

    // 默认模型路径
    private static final String DEFAULT_MODEL_PATH = Paths.get("models", "layout_cdla.onnx").toString();

    // 模型类型
    private final String modelType;
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
    private final List<String> ppLayoutType;
    private final List<String> yoloLayoutType;
    private final List<String> docLayoutType;

    /**
     * 构造函数
     * @param modelType  例如 "pp_layout_cdla", "yolov8n_layout_publaynet", "doclayout_yolo" 等
     * @param modelPath  模型文件路径
     * @param confThres  置信度阈值 (0~1)
     * @param iouThres   NMS iou阈值 (0~1)
     * @param useCuda    是否使用GPU(CUDA)
     */
    public RapidLayout(String modelType, String modelPath, float confThres, float iouThres, boolean useCuda) throws DownloadModel.DownloadModelError {
        OpencvLoader.loadOpencvLib();
        // 校验阈值
        if (!checkOf(confThres)) {
            throw new IllegalArgumentException("conf_thres " + confThres + " 超出 [0,1] 范围");
        }
        if (!checkOf(iouThres)) {
            throw new IllegalArgumentException("iou_thres " + iouThres + " 超出 [0,1] 范围");
        }
        this.modelType = modelType;

        // 确定最终模型路径 (本地 or 下载)
        String finalModelPath = getModelPath(modelType, modelPath);

        // 构建 session 配置
        OrtInferConfig config = new OrtInferConfig();
        config.setModelPath(finalModelPath);
        config.setUseCuda(useCuda);

        // 初始化 ONNXRuntime session
        this.session = new OrtInferSession(config);
        List<String> labels = this.session.getCharacterList("character");
        logger.info(modelType + " contains " + labels);

        // 初始化三种前处理 & 后处理
        this.ppPreProcess = new PPPreProcess(new Size(608, 800));
        this.ppPostProcess = new PPPostProcess(labels, confThres, iouThres);

        this.yoloPreProcess = new YOLOv8PreProcess(yoloInputShape[0], yoloInputShape[1]);
        this.yoloPostProcess = new YOLOv8PostProcess(labels, confThres, iouThres);

        this.doclayoutPreProcess  = new DocLayoutPreProcess(doclayoutShape[0], doclayoutShape[1]);
        this.doclayoutPostProcess = new DocLayoutPostProcess(labels, confThres, iouThres);

        // 加载图片的工具
        this.loadImg = new LoadImage();

        // 分组三种模型类型
        this.ppLayoutType  = new ArrayList<>();
        this.yoloLayoutType= new ArrayList<>();
        this.docLayoutType = new ArrayList<>();
        // 根据 KEY_TO_MODEL_URL 里的 key 判断归属
        for (String k : KEY_TO_MODEL_URL.keySet()) {
            if (k.startsWith("pp_")) {
                ppLayoutType.add(k);
            } else if (k.startsWith("yolov8n_")) {
                yoloLayoutType.add(k);
            } else if (k.startsWith("doclayout")) {
                docLayoutType.add(k);
            }
        }
    }

    /**
     * 供外部调用的推理接口
     * @param imgContent  图片输入(路径/字节/矩阵)
     * @return LayoutResult: { boxes, scores, classNames, elapsed }
     */
    public LayoutResult call(Object imgContent) throws Exception {
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

    /**
     * 传入 modelType/modelPath, 如果 modelPath 不为空则直接用它，
     * 否则根据 KEY_TO_MODEL_URL 下载对应的 onnx，再返回本地路径
     */
    private static String getModelPath(String modelType, String modelPath) throws DownloadModel.DownloadModelError {
        if (modelPath != null && !modelPath.isEmpty()) {
            return modelPath;
        }
        String url = KEY_TO_MODEL_URL.getOrDefault(modelType, null);
        if (url != null) {
            String localPath = DownloadModel.download(url);
            return localPath;
        }
        logger.info("model url is null, 使用默认模型 " + DEFAULT_MODEL_PATH);
        return DEFAULT_MODEL_PATH;
    }

    /**
     * 检查阈值取值合法性: [0,1]
     */
    private static boolean checkOf(float thres) {
        return thres >= 0.0f && thres <= 1.0f;
    }

}
