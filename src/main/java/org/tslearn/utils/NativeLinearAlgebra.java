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
        return eigenSystem(symmetric, size, count).vectors();
    }

    public record EigenSystem(double[][] vectors, double[] values) {}

    public static EigenSystem eigenSystem(double[] symmetric, int size, int count) {
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
        double[] values = new double[count];
        for (int i = 0; i < count; i++) values[i] = eigenvalues[size - 1 - i];
        return new EigenSystem(result, values);
    }

    /** Thin native SVD: squared singular values and right singular vectors, largest first.
     * Avoids a time-by-time covariance allocation for long, skinny datasets. */
    public static EigenSystem rightSingularSystem(double[] matrix, int rows, int cols, int count) {
        int rank=Math.min(rows,cols);
        if(rows<1 || cols<1 || count<1 || count>rank || matrix.length!=Math.multiplyExact(rows,cols))
            throw new IllegalArgumentException("Invalid SVD shape");
        double[] singular=new double[rank], vt=new double[Math.multiplyExact(rank,cols)];
        int info=LAPACKE_dgesdd(LAPACK_ROW_MAJOR,(byte)'S',rows,cols,matrix.clone(),cols,singular,
                new double[Math.multiplyExact(rows,rank)],rank,vt,cols);
        if(info!=0)throw new IllegalStateException("LAPACK dgesdd failed: info="+info);
        double[][] vectors=new double[count][cols];double[] values=new double[count];
        for(int k=0;k<count;k++) {
            System.arraycopy(vt,k*cols,vectors[k],0,cols);
            // sklearn uses deterministic signs based on each right singular vector.
            int max=0;for(int j=1;j<cols;j++)if(Math.abs(vectors[k][j])>Math.abs(vectors[k][max]))max=j;
            if(vectors[k][max]<0)for(int j=0;j<cols;j++)vectors[k][j]=-vectors[k][j];
            values[k]=singular[k]*singular[k];
        }
        return new EigenSystem(vectors,values);
    }

    /** LAPACK column-pivoted QR order, including unselected columns after numerical rank. */
    public static int[] columnPivotOrder(double[][] matrix) {
        int rows=matrix.length;
        if(rows==0 || matrix[0].length==0)throw new IllegalArgumentException("Invalid QR shape");
        int cols=matrix[0].length;double[] flat=new double[Math.multiplyExact(rows,cols)];
        for(int i=0;i<rows;i++){if(matrix[i].length!=cols)throw new IllegalArgumentException("Ragged QR matrix");System.arraycopy(matrix[i],0,flat,i*cols,cols);}
        int[] pivots=new int[cols];
        int info=LAPACKE_dgeqp3(LAPACK_ROW_MAJOR,rows,cols,flat,cols,pivots,new double[Math.min(rows,cols)]);
        if(info!=0)throw new IllegalStateException("LAPACK dgeqp3 failed: info="+info);
        for(int i=0;i<cols;i++)pivots[i]--;return pivots;
    }

    /** Minimum-norm least squares, including rank-deficient/underdetermined inputs. */
    public static double[] leastSquares(double[][] design, double[] target) {
        int rows = design.length;
        if (rows == 0 || target.length != rows || design[0].length == 0)
            throw new IllegalArgumentException("Invalid least-squares shape");
        int cols = design[0].length;
        double[] a = new double[Math.multiplyExact(rows, cols)];
        for (int i = 0; i < rows; i++) {
            if (design[i].length != cols) throw new IllegalArgumentException("Ragged design matrix");
            System.arraycopy(design[i], 0, a, i * cols, cols);
        }
        double[] b = new double[Math.max(rows, cols)];
        System.arraycopy(target, 0, b, 0, rows);
        int info = LAPACKE_dgelsd(LAPACK_ROW_MAJOR, rows, cols, 1, a, cols, b, 1,
                new double[Math.min(rows, cols)], Math.ulp(1.0) * Math.max(rows, cols), new int[1]);
        if (info != 0) throw new IllegalStateException("LAPACK dgelsd failed: info=" + info);
        return java.util.Arrays.copyOf(b, cols);
    }
}
