package org.tslearn.utils;

/**
 * Utilitários para manipulação de arrays multidimensionais.
 */
public class ArrayUtils {
    
    /**
     * Cria uma cópia profunda de um array 2D.
     */
    public static double[][] deepCopy2D(double[][] array) {
        if (array == null) return null;
        
        double[][] copy = new double[array.length][];
        for (int i = 0; i < array.length; i++) {
            if (array[i] != null) {
                copy[i] = array[i].clone();
            }
        }
        return copy;
    }
    
    /**
     * Cria uma cópia profunda de um array 3D.
     */
    public static double[][][] deepCopy3D(double[][][] array) {
        if (array == null) return null;
        
        double[][][] copy = new double[array.length][][];
        for (int i = 0; i < array.length; i++) {
            if (array[i] != null) {
                copy[i] = deepCopy2D(array[i]);
            }
        }
        return copy;
    }
    
    /**
     * Converte array 2D para matriz unidimensional (row-major order).
     */
    public static double[] flatten2D(double[][] array) {
        if (array == null || array.length == 0) return new double[0];
        
        int rows = array.length;
        int cols = array[0].length;
        double[] flattened = new double[rows * cols];
        
        for (int i = 0; i < rows; i++) {
            System.arraycopy(array[i], 0, flattened, i * cols, cols);
        }
        
        return flattened;
    }
    
    /**
     * Converte array unidimensional para array 2D.
     */
    public static double[][] unflatten2D(double[] array, int rows, int cols) {
        if (array.length != rows * cols) {
            throw new IllegalArgumentException("Array size doesn't match dimensions");
        }
        
        double[][] result = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(array, i * cols, result[i], 0, cols);
        }
        
        return result;
    }
    
    /**
     * Calcula a média de um conjunto de séries temporais.
     */
    public static double[][] mean(double[][][] timeSeries) {
        if (timeSeries == null || timeSeries.length == 0) {
            return new double[0][0];
        }
        
        int timeLength = timeSeries[0].length;
        int nFeatures = timeSeries[0][0].length;
        double[][] mean = new double[timeLength][nFeatures];
        
        for (int t = 0; t < timeLength; t++) {
            for (int d = 0; d < nFeatures; d++) {
                double sum = 0.0;
                for (int i = 0; i < timeSeries.length; i++) {
                    sum += timeSeries[i][t][d];
                }
                mean[t][d] = sum / timeSeries.length;
            }
        }
        
        return mean;
    }
}
