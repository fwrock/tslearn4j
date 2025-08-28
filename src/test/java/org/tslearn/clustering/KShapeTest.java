package org.tslearn.clustering;

import org.apache.commons.math3.linear.RealMatrix;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;
import org.tslearn.utils.MatrixUtils;

class KShapeTest {
    
    @Test
    void testKShapeBasic() {
        // Create simple test data: 3 time series of length 10
        double[][] data = {
            {1.0, 2.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 2.0},
            {2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0},
            {0.0, 1.0, 2.0, 1.0, 0.0, -1.0, 0.0, 1.0, 2.0, 1.0}
        };
        
        // Convert to RealMatrix format
        RealMatrix[] timeSeries = MatrixUtils.toTimeSeriesDataset(data);
        
        // Test KShape clustering
        KShape kshape = new KShape(2, 10, 1e-3, 1, true, 42L, "random");
        kshape.fit(timeSeries);
        
        // Basic assertions
        assertTrue(kshape.isFitted());
        assertNotNull(kshape.getLabels());
        assertEquals(3, kshape.getLabels().length);
        assertEquals(2, kshape.getClusterCenters().length);
        assertTrue(kshape.getInertia() >= 0);
        assertTrue(kshape.getNIter() >= 0);
        
        // Test prediction
        int[] labels = kshape.predict(timeSeries);
        assertNotNull(labels);
        assertEquals(3, labels.length);
        
        // All labels should be 0 or 1 (since we have 2 clusters)
        for (int label : labels) {
            assertTrue(label >= 0 && label < 2);
        }
    }
    
    @Test
    void testKShapeWithDoubleArray() {
        // Test convenience method with double array
        double[][] data = {
            {1.0, 2.0, 3.0, 2.0, 1.0},
            {2.0, 3.0, 4.0, 3.0, 2.0},
            {0.0, 1.0, 2.0, 1.0, 0.0}
        };
        
        KShape kshape = new KShape(2, 5, 1e-3, 1, false, 42L, "random");
        kshape.fit(data);
        
        assertTrue(kshape.isFitted());
        int[] labels = kshape.predict(data);
        assertNotNull(labels);
        assertEquals(3, labels.length);
    }
    
    @Test
    void testKShapeParameters() {
        KShape kshape = new KShape();
        
        // Test default parameters
        assertEquals(3, kshape.getNClusters());
        assertEquals(100, kshape.getMaxIter());
        assertEquals(1e-6, kshape.getTol(), 1e-9);
        assertEquals(1, kshape.getNInit());
        assertFalse(kshape.isVerbose());
        assertEquals(0L, kshape.getRandomState());
    }
    
    @Test
    void testEmptyDataException() {
        KShape kshape = new KShape();
        
        assertThrows(IllegalArgumentException.class, () -> {
            kshape.fit(new RealMatrix[0]);
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            kshape.fit((RealMatrix[]) null);
        });
    }
    
    @Test
    void testPredictBeforeFit() {
        KShape kshape = new KShape();
        double[][] data = {{1.0, 2.0, 3.0}};
        
        assertThrows(IllegalStateException.class, () -> {
            kshape.predict(data);
        });
    }
}
