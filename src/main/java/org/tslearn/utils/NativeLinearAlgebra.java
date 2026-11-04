package org.tslearn.utils;

import static org.bytedeco.openblas.global.openblas.*;

/** Double-precision OpenBLAS/LAPACK operations on contiguous row-major arrays. */
public final class NativeLinearAlgebra {
    private NativeLinearAlgebra() {}

    /** Return X^T X without creating a transposed copy of X. */
    public static double[] gram(double[] x, int rows, int columns) {
        if (rows < 1 || columns < 1 || x.length != Math.multiplyExact(rows, columns)) {
            throw new IllegalArgumentException("Invalid matrix shape");
        }
        double[] result = new double[Math.multiplyExact(columns, columns)];
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                columns, columns, rows, 1.0, x, columns, x, columns, 0.0, result, columns);
        return result;
    }

    /** Largest eigenvectors first, each returned as a row. The input is preserved. */
    public static double[][] leadingEigenvectors(double[] symmetric, int size, int count) {
        if (size < 1 || count < 1 || count > size || symmetric.length != Math.multiplyExact(size, size)) {
            throw new IllegalArgumentException("Invalid eigensystem shape");
        }
        double[] vectors = symmetric.clone();
        double[] eigenvalues = new double[size];
        int info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, (byte) 'V', (byte) 'U', size, vectors, size, eigenvalues);
        if (info != 0) {
            throw new IllegalStateException("LAPACK dsyev failed: info=" + info);
        }
        double[][] result = new double[count][size];
        for (int component = 0; component < count; component++) {
            for (int row = 0; row < size; row++) {
                result[component][row] = vectors[row * size + size - 1 - component];
            }
        }
        return result;
    }
}
