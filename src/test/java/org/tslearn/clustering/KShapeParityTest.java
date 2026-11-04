package org.tslearn.clustering;

import org.junit.jupiter.api.Test;
import org.tslearn.preprocessing.TimeSeriesScalerMeanVariance;
import org.tslearn.utils.MatrixUtils;
import org.tslearn.utils.EmptyClusterException;
import static org.junit.jupiter.api.Assertions.*;

class KShapeParityTest {
    @Test void populationVarianceAndIndependentNormalization() {
        var input = MatrixUtils.toTimeSeriesDataset(new double[][]{{1,2,3,4},{101,103,105,107}});
        var scaled = new TimeSeriesScalerMeanVariance().fitTransform(input);
        double[] expected = {-1.3416407864998738,-.4472135954999579,.4472135954999579,1.3416407864998738};
        assertArrayEquals(expected, MatrixUtils.matrixToArray(scaled[0]), 1e-12);
        assertArrayEquals(expected, MatrixUtils.matrixToArray(scaled[1]), 1e-12);
    }
    @Test void predictionsAreInvariantToPerSeriesOffsetAndPositiveScale() {
        double[][] data = {{-1,0,1,0,-1,0,1,0},{0,1,0,-1,0,1,0,-1},
                           {-1,-1,-1,-1,1,1,1,1},{1,1,1,1,-1,-1,-1,-1}};
        var init = MatrixUtils.toTimeSeriesDataset(new double[][]{data[0],data[2]});
        var model = new KShape(2,10,1e-6,1,false,42,init).fit(data);
        double[][] transformed = new double[data.length][data[0].length];
        for(int i=0;i<data.length;i++) for(int j=0;j<data[i].length;j++) transformed[i][j]=data[i][j]*(i+1)+100*i;
        assertArrayEquals(model.predict(data), model.predict(transformed));
    }
    @Test void impossibleClusteringDoesNotReturnFittedModelWithInfiniteInertia() {
        var model = new KShape(2,10,1e-6,1,false,42,"random");
        assertThrows(EmptyClusterException.class, () -> model.fit(new double[][]{{1,1,1},{1,1,1},{1,1,1}}));
        assertFalse(model.isFitted());
    }
    @Test void inertiaUsesMeanSquaredDistances() {
        assertEquals(.125, MatrixUtils.computeInertia(new double[][]{{.5,.1},{.2,0}}, new int[]{0,1}), 1e-12);
    }
}
