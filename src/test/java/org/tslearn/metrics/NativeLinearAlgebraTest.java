package org.tslearn.metrics;

import org.junit.jupiter.api.Test;
import org.tslearn.utils.NativeLinearAlgebra;
import static org.junit.jupiter.api.Assertions.*;

class NativeLinearAlgebraTest {
    @Test void covarianceAndEigenvectorUseCorrectRowMajorOrientation() {
        double[] gram = NativeLinearAlgebra.gram(new double[]{1,2,3,4,5,6},3,2);
        assertArrayEquals(new double[]{35,44,44,56},gram,1e-12);
        double[] input = {2,1,1,2};
        double[][] components = NativeLinearAlgebra.leadingEigenvectors(input,2,2);
        for(int row=0;row<2;row++) {
            double eigenvalue = row==0?3:1;
            double[] v=components[row];
            assertEquals(eigenvalue*v[0],2*v[0]+v[1],1e-12);
            assertEquals(eigenvalue*v[1],v[0]+2*v[1],1e-12);
            assertEquals(1,v[0]*v[0]+v[1]*v[1],1e-12);
        }
        assertArrayEquals(new double[]{2,1,1,2},input);
    }
    @Test void invalidDimensionsFailBeforeCallingNativeCode() {
        assertThrows(IllegalArgumentException.class,()->NativeLinearAlgebra.gram(new double[3],2,2));
        assertThrows(IllegalArgumentException.class,()->NativeLinearAlgebra.leadingEigenvectors(new double[4],2,3));
    }
    @Test void shortAndFftSeriesUseTheSameShiftOrientation() {
        for (int length : new int[]{32,180}) {
            double[] reference=new double[length];
            double[] shifted=new double[length];
            reference[5]=1; reference[6]=2; reference[7]=1;
            shifted[8]=1; shifted[9]=2; shifted[10]=1;
            var ref=org.tslearn.utils.MatrixUtils.arrayToMatrix(reference);
            var input=org.tslearn.utils.MatrixUtils.toTimeSeriesDataset(new double[][]{shifted});
            var result=CrossCorrelation.yShiftedSbdVec(ref,input,-1,new double[]{Math.sqrt(6)});
            assertArrayEquals(reference,org.tslearn.utils.MatrixUtils.matrixToArray(result[0]),1e-12);
        }
    }
}
