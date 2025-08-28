package org.tslearn.clustering;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/**
 * Testes unitários para TimeSeriesKMeans.
 */
public class TimeSeriesKMeansTest {
    
    private double[][][] testData;
    
    @BeforeEach
    void setUp() {
        // Criar dataset simples para testes
        testData = new double[6][5][1];
        
        // Cluster 1: valores próximos de 1
        testData[0] = new double[][]{{1.0}, {1.1}, {0.9}, {1.0}, {1.1}};
        testData[1] = new double[][]{{1.1}, {1.0}, {1.2}, {0.9}, {1.0}};
        
        // Cluster 2: valores próximos de 5
        testData[2] = new double[][]{{5.0}, {5.1}, {4.9}, {5.0}, {5.1}};
        testData[3] = new double[][]{{5.1}, {5.0}, {5.2}, {4.9}, {5.0}};
        
        // Cluster 3: valores próximos de 10
        testData[4] = new double[][]{{10.0}, {10.1}, {9.9}, {10.0}, {10.1}};
        testData[5] = new double[][]{{10.1}, {10.0}, {10.2}, {9.9}, {10.0}};
    }
    
    @Test
    void testEuclideanKMeans() {
        TimeSeriesKMeans kmeans = new TimeSeriesKMeans.Builder()
                .nClusters(3)
                .metric(TimeSeriesKMeans.Metric.EUCLIDEAN)
                .maxIter(100)
                .nInit(1)
                .randomSeed(42)
                .build();
        
        kmeans.fit(testData);
        
        assertTrue(kmeans.isFitted());
        assertEquals(3, kmeans.getNumClusters());
        assertNotNull(kmeans.getLabels());
        assertNotNull(kmeans.getClusterCenters());
        assertTrue(kmeans.getInertia() >= 0);
        assertTrue(kmeans.getNumIterations() > 0);
        
        // Verificar que conseguiu separar os clusters
        int[] labels = kmeans.getLabels();
        assertEquals(6, labels.length);
        
        // Os primeiros dois devem estar no mesmo cluster
        assertEquals(labels[0], labels[1]);
        // Os do meio devem estar no mesmo cluster
        assertEquals(labels[2], labels[3]);
        // Os últimos devem estar no mesmo cluster
        assertEquals(labels[4], labels[5]);
    }
    
    @Test
    void testDTWKMeans() {
        TimeSeriesKMeans kmeans = new TimeSeriesKMeans.Builder()
                .nClusters(3)
                .metric(TimeSeriesKMeans.Metric.DTW)
                .maxIter(50)
                .maxIterBarycenter(10)
                .nInit(1)
                .randomSeed(42)
                .build();
        
        kmeans.fit(testData);
        
        assertTrue(kmeans.isFitted());
        assertEquals(TimeSeriesKMeans.Metric.DTW, kmeans.getMetric());
        assertNotNull(kmeans.getLabels());
        assertTrue(kmeans.getInertia() >= 0);
    }
    
    @Test
    void testPredict() {
        TimeSeriesKMeans kmeans = new TimeSeriesKMeans.Builder()
                .nClusters(3)
                .metric(TimeSeriesKMeans.Metric.EUCLIDEAN)
                .nInit(1)
                .randomSeed(42)
                .build();
        
        kmeans.fit(testData);
        
        // Testar predição com novos dados
        double[][][] newData = new double[2][5][1];
        newData[0] = new double[][]{{1.05}, {1.0}, {0.95}, {1.1}, {1.0}};  // Próximo ao cluster 1
        newData[1] = new double[][]{{9.9}, {10.0}, {10.1}, {9.8}, {10.2}}; // Próximo ao cluster 3
        
        int[] predictions = kmeans.predict(newData);
        assertEquals(2, predictions.length);
        
        // Verificar que as predições fazem sentido
        assertTrue(predictions[0] >= 0 && predictions[0] < 3);
        assertTrue(predictions[1] >= 0 && predictions[1] < 3);
    }
    
    @Test
    void testFitPredict() {
        TimeSeriesKMeans kmeans = new TimeSeriesKMeans.Builder()
                .nClusters(3)
                .metric(TimeSeriesKMeans.Metric.EUCLIDEAN)
                .nInit(1)
                .randomSeed(42)
                .build();
        
        int[] labels = kmeans.fitPredict(testData);
        
        assertTrue(kmeans.isFitted());
        assertEquals(6, labels.length);
        assertArrayEquals(labels, kmeans.getLabels());
    }
    
    @Test
    void testTransform() {
        TimeSeriesKMeans kmeans = new TimeSeriesKMeans.Builder()
                .nClusters(3)
                .metric(TimeSeriesKMeans.Metric.EUCLIDEAN)
                .nInit(1)
                .randomSeed(42)
                .build();
        
        kmeans.fit(testData);
        
        double[][] distances = kmeans.transform(testData);
        assertEquals(6, distances.length);
        assertEquals(3, distances[0].length);
        
        // Verificar que todas as distâncias são não-negativas
        for (int i = 0; i < distances.length; i++) {
            for (int j = 0; j < distances[i].length; j++) {
                assertTrue(distances[i][j] >= 0);
            }
        }
    }
    
    @Test
    void testInvalidInputs() {
        TimeSeriesKMeans kmeans = new TimeSeriesKMeans.Builder()
                .nClusters(3)
                .build();
        
        // Testar com dados nulos
        assertThrows(IllegalArgumentException.class, () -> {
            kmeans.fit(null);
        });
        
        // Testar com dados vazios
        assertThrows(IllegalArgumentException.class, () -> {
            kmeans.fit(new double[0][][]);
        });
        
        // Testar com mais clusters que amostras
        assertThrows(IllegalArgumentException.class, () -> {
            TimeSeriesKMeans tooManyClusters = new TimeSeriesKMeans.Builder()
                    .nClusters(10)
                    .build();
            tooManyClusters.fit(testData);
        });
    }
    
    @Test
    void testNotFittedState() {
        TimeSeriesKMeans kmeans = new TimeSeriesKMeans.Builder()
                .nClusters(3)
                .build();
        
        assertFalse(kmeans.isFitted());
        assertNull(kmeans.getClusterCenters());
        assertNull(kmeans.getLabels());
        
        // Deve lançar exceção se tentar predizer sem treinar
        assertThrows(IllegalStateException.class, () -> {
            kmeans.predict(testData);
        });
        
        assertThrows(IllegalStateException.class, () -> {
            kmeans.transform(testData);
        });
    }
    
    @Test
    void testBuilderPattern() {
        TimeSeriesKMeans kmeans = new TimeSeriesKMeans.Builder()
                .nClusters(5)
                .maxIter(200)
                .tolerance(1e-8)
                .nInit(5)
                .metric(TimeSeriesKMeans.Metric.DTW)
                .maxIterBarycenter(50)
                .verbose(true)
                .dtwInertia(true)
                .randomSeed(123)
                .metricParam("sakoeChiba", 10)
                .build();
        
        assertEquals(5, kmeans.getNumClusters());
        assertEquals(TimeSeriesKMeans.Metric.DTW, kmeans.getMetric());
    }
}
