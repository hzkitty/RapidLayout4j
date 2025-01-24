package io.github.hzkitty.rapid_layout.utils;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import org.opencv.core.Mat;

import java.nio.FloatBuffer;

public class NDArrayUtils {

    /**
     * 将任意类型的多维数组转换为一维数组并返回Shape
     * @param arr 任意类型的多维数组
     * @return 一个包含一维数组数据和Shape对象的数组
     */
    public static NDArray create(NDManager manager, float[][][] arr) {
        int dim1 = arr.length;
        int dim2 = arr[0].length;
        int dim3 = arr[0][0].length;
        float[] ret = new float[dim1 * dim2 * dim3];
        int count = 0;
        for(int i = 0; i < arr.length; ++i) {
            for(int j = 0; j < arr[0].length; ++j) {
                System.arraycopy(arr[i][j], 0, ret, count, arr[0][0].length);
                count += arr[0][0].length;
            }
        }
        return manager.create(ret, new Shape(dim1, dim2, dim3));
    }

    public static NDArray create(NDManager manager, float[][][][] arr) {
        int dim1 = arr.length;
        int dim2 = arr[0].length;
        int dim3 = arr[0][0].length;
        int dim4 = arr[0][0][0].length;
        float[] ret = new float[dim1 * dim2 * dim3 * dim4];
        int count = 0;

        for(int i = 0; i < arr.length; ++i) {
            for(int j = 0; j < arr[0].length; ++j) {
                for(int k = 0; k < arr[0][0].length; ++k) {
                    System.arraycopy(arr[i][j][k], 0, ret, count, arr[0][0][0].length);
                    count += arr[0][0][0].length;
                }
            }
        }
        return manager.create(ret, new Shape(dim1, dim2, dim3, dim4));
    }

    /**
     * 将 NDArray 转换为三维数组（double[][][]）
     *
     * @param array 输入的 NDArray
     * @return 转换后的三维数组
     */
    public static float[][][] toFloatArray3D(NDArray array) {
        Shape shape = array.getShape();
        if (shape.dimension() != 3) {
            throw new IllegalArgumentException("输入的 NDArray 必须是三维的");
        }
        long dim1 = shape.get(0);
        long dim2 = shape.get(1);
        long dim3 = shape.get(2);

        float[] flatArray = array.toFloatArray();
        float[][][] result = new float[(int) dim1][(int) dim2][(int) dim3];
        int index = 0;
        for (int i = 0; i < dim1; i++) {
            for (int j = 0; j < dim2; j++) {
                for (int k = 0; k < dim3; k++) {
                    result[i][j][k] = flatArray[index++];
                }
            }
        }
        return result;
    }

    public static float[][][][] toFloatArray4D(NDArray array) {
        Shape shape = array.getShape();
        if (shape.dimension() != 4) {
            throw new IllegalArgumentException("输入的 NDArray 必须是四维的");
        }
        long dim1 = shape.get(0);
        long dim2 = shape.get(1);
        long dim3 = shape.get(2);
        long dim4 = shape.get(3);
        float[] flatArray = array.toFloatArray();
        float[][][][] result = new float[(int) dim1][(int) dim2][(int) dim3][(int) dim4];
        int index = 0;
        for (int i = 0; i < dim1; i++) {
            for (int j = 0; j < dim2; j++) {
                for (int k = 0; k < dim3; k++) {
                    for (int l = 0; l < dim4; l++) {
                        result[i][j][k][l] = flatArray[index++];
                    }
                }
            }
        }
        return result;
    }

    public static NDArray create(NDManager manager, Mat mat) {
        float[] buf = new float[mat.height() * mat.width() * mat.channels()];
        mat.get(0, 0, buf);
        Shape shape = new Shape(mat.height(), mat.width(), mat.channels());
        return manager.create(FloatBuffer.wrap(buf), shape, DataType.FLOAT32);
    }
}
