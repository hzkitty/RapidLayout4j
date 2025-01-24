package io.github.hzkitty.rapid_layout.utils;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * DownloadModel 负责从指定 URL 下载文件到本地
 */
public class DownloadModel {

    private static final Logger logger = Logger.getLogger(DownloadModel.class.getName());

    private static final Path PROJECT_DIR = Paths.get(System.getProperty("user.dir")).toAbsolutePath();

    /**
     * 下载主方法
     *
     * @param modelFullUrl 远程文件地址（包含文件名）
     * @return 下载后的本地文件绝对路径
     * @throws DownloadModelError 如果下载失败，会抛出自定义异常
     */
    public static String download(String modelFullUrl) throws DownloadModelError {
        // 在当前项目目录下，创建一个 models 文件夹，用于存储下载的文件
        Path saveDir = PROJECT_DIR.resolve("models");
        try {
            if (!Files.exists(saveDir)) {
                Files.createDirectories(saveDir);
            }
        } catch (Exception e) {
            logger.log(Level.SEVERE, "创建文件夹失败: " + saveDir, e);
            throw new DownloadModelError("无法创建下载目录: " + saveDir, e);
        }

        // 获取需要下载的文件名
        String modelName = modelFullUrl.substring(modelFullUrl.lastIndexOf('/') + 1);;
        Path saveFilePath = saveDir.resolve(modelName);

        // 如果目标文件已存在，直接返回
        if (Files.exists(saveFilePath)) {
            logger.info(saveFilePath + " 已经存在，跳过下载。");
            return saveFilePath.toString();
        }

        // 如果文件不存在，则进行下载
        try {
            logger.info(String.format("开始下载 %s 到 %s", modelFullUrl, saveDir.toString()));
            byte[] fileBytes = downloadAsBytesWithProgress(modelFullUrl, modelName);
            saveFile(saveFilePath, fileBytes);
            logger.info("下载完成，文件保存在: " + saveFilePath.toAbsolutePath());
        } catch (Exception e) {
            logger.log(Level.SEVERE, "下载失败: " + modelFullUrl, e);
            throw new DownloadModelError("下载失败: " + modelFullUrl, e);
        }

        // 返回下载后的文件路径
        return saveFilePath.toString();
    }

    /**
     * 从指定 URL 下载文件字节，带有简单的进度显示
     *
     * @param url  下载链接
     * @param name 用于显示进度时的文件名
     * @return 完整的文件字节数组
     * @throws Exception 任何 IO 或网络异常都可能抛出
     */
    private static byte[] downloadAsBytesWithProgress(String url, String name) throws Exception {
        HttpURLConnection connection = null;
        try {
            // 打开连接
            connection = (HttpURLConnection) new URL(url).openConnection();
            // 设置超时时间，可根据需要调整
            connection.setConnectTimeout(20_000);
            connection.setReadTimeout(180_000);
            connection.setRequestMethod("GET");

            // 获取文件总大小，用于计算进度
            int contentLength = connection.getContentLength();
            if (contentLength < 0) {
                logger.warning("无法获取文件大小，进度显示可能不准确。");
            }

            // 尝试读取数据
            try (InputStream in = connection.getInputStream();
                 ByteArrayOutputStream out = new ByteArrayOutputStream()) {

                byte[] buffer = new byte[65536];
                int bytesRead;
                long totalRead = 0;

                // 读取并写入 ByteArrayOutputStream
                while ((bytesRead = in.read(buffer)) != -1) {
                    out.write(buffer, 0, bytesRead);
                    totalRead += bytesRead;

                    // 如果文件大小可用，则进行进度计算
                    if (contentLength > 0) {
                        double progress = (double) totalRead / contentLength * 100;
                        // 这里简单打印进度，如有需要可使用更复杂的进度条或日志
                        System.out.printf("\r[%s] 已下载: %.2f%%", name, progress);
                    }
                }

                // 下载完后换行，防止覆盖进度信息
                System.out.println();
                return out.toByteArray();
            }

        } finally {
            // 断开连接
            if (connection != null) {
                connection.disconnect();
            }
        }
    }

    /**
     * 保存字节数组到本地文件
     *
     * @param savePath 目标文件路径
     * @param file     文件字节数组
     * @throws Exception 任何 IO 异常都会抛出
     */
    private static void saveFile(Path savePath, byte[] file) throws Exception {
        File targetFile = savePath.toFile();
        // 尝试创建任何不存在的父目录
        targetFile.getParentFile().mkdirs();

        try (FileOutputStream fos = new FileOutputStream(targetFile)) {
            fos.write(file);
        }
    }

    /**
     * 自定义下载异常类，可根据需要扩展
     */
    public static class DownloadModelError extends Exception {
        public DownloadModelError(String message) {
            super(message);
        }

        public DownloadModelError(String message, Throwable cause) {
            super(message, cause);
        }
    }

}
