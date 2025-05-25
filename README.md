<div align="center">
  <div align="center">
    <h1><b>📖 Rapid Layout4j</b></h1>
  </div>
</div>

### 简介

RapidLayout主要是汇集全网开源的版面分析的项目，具体来说，就是分析给定的文档类别图像（论文截图、研报等），定位其中类别和位置，如标题、段落、表格和图片等各个部分。

本项目是RapidLayout的Java移植版本，使用 ONNXRuntime + OpenCV + NDArray。

⚠️注意：需要说明的是，由于不同场景下的版面差异较大，现阶段不存在一个模型可以搞定所有场景。如果实际业务需要，以下模型效果不好的话，建议构建自己的训练集微调。

目前支持已经支持的版面分析模型如下：

|`model_type`| 版面类型 |  支持类别|
| :------ | :----- | :----- |
|`pp_layout_table`|   表格   |`["table"]` |
| `pp_layout_publaynet`|   英文   |`["text", "title", "list", "table", "figure"]` |
| `pp_layout_cdla`|   中文    | `['text', 'title', 'figure', 'figure_caption', 'table', 'table_caption', 'header', 'footer', 'reference', 'equation']` |
| `yolov8n_layout_paper`|   论文    | `['Text', 'Title', 'Header', 'Footer', 'Figure', 'Table', 'Toc', 'Figure caption', 'Table caption']` |
| `yolov8n_layout_report`|   研报    | `['Text', 'Title', 'Header', 'Footer', 'Figure', 'Table', 'Toc', 'Figure caption', 'Table caption']` |
| `yolov8n_layout_publaynet`|   英文     | `["Text", "Title", "List", "Table", "Figure"]` |
| `yolov8n_layout_general6`|   通用      | `["Text", "Title", "Figure", "Table", "Caption", "Equation"]` |
| 🔥`doclayout_docstructbench`|   通用   | `['title', 'plain text', 'abandon', 'figure', 'figure_caption', 'table', 'table_caption', 'table_footnote', 'isolate_formula', 'formula_caption']` |
| 🔥`doclayout_d4la`|   通用       | `['DocTitle', 'ParaTitle', 'ParaText', 'ListText', 'RegionTitle', 'Date', 'LetterHead', 'LetterDear', 'LetterSign', 'Question', 'OtherText', 'RegionKV', 'RegionList', 'Abstract', 'Author', 'TableName', 'Table', 'Figure', 'FigureName', 'Equation', 'Reference', 'Footer', 'PageHeader', 'PageFooter', 'Number', 'Catalog', 'PageNumber']` |
| 🔥`doclayout_docsynth`|   通用    | `['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title']` |

PP模型来源：[PaddleOCR 版面分析](https://github.com/PaddlePaddle/PaddleOCR/blob/133d67f27dc8a241d6b2e30a9f047a0fb75bebbe/ppstructure/layout/README_ch.md)

yolov8n系列来源：[360LayoutAnalysis](https://github.com/360AILAB-NLP/360LayoutAnalysis)

（推荐使用）🔥doclayout_yolo模型来源：[DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO)，该模型是目前最为优秀的开源模型，挑选了3个基于不同训练集训练得到的模型。其中`doclayout_docstructbench`来自[link](https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench/tree/main)，`doclayout_d4la`来自[link](https://huggingface.co/juliozhao/DocLayout-YOLO-D4LA-Docsynth300K_pretrained/blob/main/doclayout_yolo_d4la_imgsz1600_docsynth_pretrain.pt)，`doclayout_docsynth`来自[link](https://huggingface.co/juliozhao/DocLayout-YOLO-DocLayNet-Docsynth300K_pretrained/tree/main)。

上述模型下载地址为：[link](https://github.com/hzkitty/RapidLayout4j/releases/tag/v0.0.0)


### 安装

由于模型较小，预先将中文版面分析模型(`layout_cdla.onnx`)打包进了jar包内。其余模型在初始化`RapidLayout`类时，通过`LayoutConfig的modelPath`来指定自己模型路径。注意仅限于现在支持的`LayoutModelType`。

## 🎉 快速开始

安装依赖，默认使用CPU版本
```xml
<dependency>
    <groupId>io.github.hzkitty</groupId>
    <artifactId>rapid-layout4j</artifactId>
    <version>1.0.0</version>
</dependency>
```
使用示例
```java
RapidLayout rapidLayout = RapidLayout.create();
File file = new File("src/test/resources/layout.png");
String imgContent = file.getAbsolutePath();
LayoutResult layoutResult = rapidLayout.run(imgContent);
```

如果想要使用GPU, `onnxruntime_gpu` 对应版本可以在这里找到
[here](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).
```xml
<dependency>
    <groupId>io.github.hzkitty</groupId>
    <artifactId>rapid-layout4j</artifactId>
    <version>1.0.0</version>
    <exclusions>
      <exclusion>
        <groupId>com.microsoft.onnxruntime</groupId>
        <artifactId>onnxruntime</artifactId>
      </exclusion>
    </exclusions>
</dependency>

<!-- 1.18.0 support CUDA 12.x -->
<dependency>
    <groupId>com.microsoft.onnxruntime</groupId>
    <artifactId>onnxruntime_gpu</artifactId>
    <version>1.18.0</version>
</dependency>
```

移植RapidLayout时，由于涉及到很多复杂的numpy操作，用普通java多维数组实现困难，所以使用到了ai.djl.ndarray.NDArray进行转换。\
目前使用了 PyTorch 引擎，进行复杂numpy操作，文档参考[DJL - PyTorch](https://docs.djl.ai/master/engines/pytorch/pytorch-engine/index.html#overview)
默认会根据当前系统自动下载pytorch引擎文件，或者可以直接导入pytorch-native-cpu包
```
<dependency>
    <groupId>ai.djl.pytorch</groupId>
    <artifactId>pytorch-native-cpu</artifactId>
    <classifier>win-x86_64</classifier>
     <!--linux-->
<!-- <classifier>linux-x86_64</classifier>-->
    <scope>runtime</scope>
    <version>2.2.2</version>
</dependency>
```

## 鸣谢

- [RapidLayout](https://github.com/RapidAI/RapidLayout)

## 开源许可
使用 [Apache License 2.0](https://github.com/MyMonsterCat/DeviceTouch/blob/main/LICENSE)
