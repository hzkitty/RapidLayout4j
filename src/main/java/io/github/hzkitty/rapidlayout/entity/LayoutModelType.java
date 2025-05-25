package io.github.hzkitty.rapidlayout.entity;

public enum LayoutModelType {

    PP_LAYOUT_TABLE("pp_layout_table"),
    PP_LAYOUT_PUBLAYNET("pp_layout_publaynet"),
    PP_LAYOUT_CDLA("pp_layout_cdla"),

    YOLOV8N_LAYOUT_PAPER("yolov8n_layout_paper"),
    YOLOV8N_LAYOUT_REPORT("yolov8n_layout_report"),
    YOLOV8N_LAYOUT_PUBLAYNET("yolov8n_layout_publaynet"),
    YOLOV8N_LAYOUT_GENERAL6("yolov8n_layout_general6"),

    DOCLAYOUT_DOCSTRUCTBENCH("doclayout_docstructbench"),
    DOCLAYOUT_D4LA("doclayout_d4la"),
    DOCLAYOUT_DOCSYNTH("doclayout_docsynth"),;

    private final String modelName;

    LayoutModelType(String modelName) {
        this.modelName = modelName;
    }

    public String getModelName() {
        return modelName;
    }

}
