package io.github.hzkitty.rapidlayout.entity;

public class LayoutConfig {

    public String modelPath = "models/layout_cdla.onnx"; // 模型路径
    public LayoutModelType modelType = LayoutModelType.PP_LAYOUT_CDLA; // 模型类型
    public boolean useCuda = false; // 是否使用 CUDA
    public int deviceId = 0; // 显卡编号
    public boolean useArena = false; // arena内存池的扩展策略（速度有提升，但内存会剧增，且持续占用，不释放，默认关闭）

    public float confThres = 0.5f; // 置信度阈值 (0~1)
    public float iouThres = 0.5f; // NMS iou阈值 (0~1)

    public String getModelPath() {
        return modelPath;
    }

    public void setModelPath(String modelPath) {
        this.modelPath = modelPath;
    }

    public LayoutModelType getModelType() {
        return modelType;
    }

    public void setModelType(LayoutModelType modelType) {
        this.modelType = modelType;
    }

    public boolean isUseCuda() {
        return useCuda;
    }

    public void setUseCuda(boolean useCuda) {
        this.useCuda = useCuda;
    }

    public int getDeviceId() {
        return deviceId;
    }

    public void setDeviceId(int deviceId) {
        this.deviceId = deviceId;
    }

    public boolean isUseArena() {
        return useArena;
    }

    public void setUseArena(boolean useArena) {
        this.useArena = useArena;
    }

    public float getConfThres() {
        return confThres;
    }

    public void setConfThres(float confThres) {
        this.confThres = confThres;
    }

    public float getIouThres() {
        return iouThres;
    }

    public void setIouThres(float iouThres) {
        this.iouThres = iouThres;
    }
}
