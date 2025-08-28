package org.tslearn.shapelets;

import java.util.Arrays;
import java.util.Random;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/**
 * Testes unitários para a implementação de Shapelets.
 */
public class ShapeletTest {
    
    private double[][][] testData;
    private String[] testLabels;
    private Random random;
    
    @BeforeEach
    void setUp() {
        random = new Random(42);
        generateTestData();
    }
    
    private void generateTestData() {
        int nSamples = 30;
        int timeLength = 50;
        int nFeatures = 1;
        
        testData = new double[nSamples][timeLength][nFeatures];
        testLabels = new String[nSamples];
        
        // Padrões discriminativos
        double[][] patternA = {{1.0}, {2.0}, {1.0}};
        double[][] patternB = {{-1.0}, {-2.0}, {-1.0}};
        
        for (int i = 0; i < nSamples; i++) {
            String label = (i % 2 == 0) ? "ClassA" : "ClassB";
            testLabels[i] = label;
            
            // Preencher com ruído
            for (int t = 0; t < timeLength; t++) {
                testData[i][t][0] = random.nextGaussian() * 0.3;
            }
            
            // Inserir padrão discriminativo
            double[][] pattern = label.equals("ClassA") ? patternA : patternB;
            int insertPos = random.nextInt(timeLength - pattern.length + 1);
            
            for (int j = 0; j < pattern.length; j++) {
                testData[i][insertPos + j][0] = pattern[j][0] + random.nextGaussian() * 0.1;
            }
        }
    }
    
    @Test
    void testShapeletCreation() {
        double[][] values = {{1.0}, {2.0}, {3.0}};
        Shapelet shapelet = new Shapelet(values, 0, 10, 0.85, "TestClass");
        
        assertEquals(3, shapelet.getLength());
        assertEquals(0, shapelet.getSourceIndex());
        assertEquals(10, shapelet.getStartPosition());
        assertEquals(0.85, shapelet.getQualityScore(), 1e-6);
        assertEquals("TestClass", shapelet.getLabel());
        assertArrayEquals(values, shapelet.getValues());
    }
    
    @Test
    void testShapeletDistance() {
        double[][] shapeletValues = {{1.0}, {2.0}, {1.0}};
        Shapelet shapelet = new Shapelet(shapeletValues, -1, -1, 0.0, "Test");
        
        // Teste com série idêntica
        double[][] identicalSeries = {{1.0}, {2.0}, {1.0}};
        double distance = shapelet.distance(identicalSeries, 0);
        assertEquals(0.0, distance, 1e-6);
        
        // Teste com série diferente
        double[][] differentSeries = {{2.0}, {3.0}, {2.0}};
        distance = shapelet.distance(differentSeries, 0);
        assertTrue(distance > 0);
    }
    
    @Test
    void testShapeletFindBestMatch() {
        double[][] shapeletValues = {{1.0}, {2.0}, {1.0}};
        Shapelet shapelet = new Shapelet(shapeletValues, -1, -1, 0.0, "Test");
        
        double[][] timeSeries = {
            {0.5}, {1.0}, {2.0}, {1.0}, {0.3}, {0.8}
        };
        
        Shapelet.ShapeletMatch match = shapelet.findBestMatch(timeSeries);
        
        assertNotNull(match);
        assertTrue(match.getDistance() >= 0);
        assertTrue(match.getPosition() >= 0);
        assertTrue(match.getPosition() <= timeSeries.length - shapeletValues.length);
    }
    
    @Test
    void testShapeletTransform() {
        double[][] shapeletValues = {{1.0}, {2.0}, {1.0}};
        Shapelet shapelet = new Shapelet(shapeletValues, -1, -1, 0.0, "Test");
        
        double[][][] dataset = {
            {{0.5}, {1.0}, {2.0}, {1.0}, {0.3}},
            {{-1.0}, {-2.0}, {-1.0}, {0.5}, {1.0}}
        };
        
        double[] transformed = shapelet.transform(dataset);
        
        assertEquals(dataset.length, transformed.length);
        for (double distance : transformed) {
            assertTrue(distance >= 0);
        }
    }
    
    @Test
    void testShapeletNormalization() {
        double[][] values = {{1.0}, {3.0}, {2.0}};
        Shapelet shapelet = new Shapelet(values, -1, -1, 0.0, "Test");
        
        Shapelet normalized = shapelet.normalize();
        
        assertNotNull(normalized);
        assertEquals(shapelet.getLength(), normalized.getLength());
        
        // Verificar que os valores foram normalizados (média 0, std 1)
        double[][] normalizedValues = normalized.getValues();
        double mean = Arrays.stream(normalizedValues)
                .mapToDouble(arr -> arr[0])
                .average().orElse(0.0);
        
        assertEquals(0.0, mean, 1e-10);
    }
    
    @Test
    void testShapeletSimilarity() {
        double[][] values1 = {{1.0}, {2.0}, {3.0}};
        double[][] values2 = {{1.1}, {2.1}, {3.1}};
        double[][] values3 = {{5.0}, {6.0}, {7.0}};
        
        Shapelet shapelet1 = new Shapelet(values1, -1, -1, 0.0, "Test");
        Shapelet shapelet2 = new Shapelet(values2, -1, -1, 0.0, "Test");
        Shapelet shapelet3 = new Shapelet(values3, -1, -1, 0.0, "Test");
        
        // Shapelets similares
        assertTrue(shapelet1.isSimilarTo(shapelet2, 0.5));
        
        // Shapelets diferentes
        assertFalse(shapelet1.isSimilarTo(shapelet3, 0.5));
    }
    
    @Test
    void testShapeletExtraction() {
        double[][] timeSeries = {
            {1.0}, {2.0}, {3.0}, {4.0}, {5.0}
        };
        
        Shapelet extracted = Shapelet.extractSubsequence(timeSeries, 1, 3);
        
        assertEquals(3, extracted.getLength());
        assertEquals(2.0, extracted.getValues()[0][0], 1e-6);
        assertEquals(3.0, extracted.getValues()[1][0], 1e-6);
        assertEquals(4.0, extracted.getValues()[2][0], 1e-6);
    }
    
    @Test
    void testShapeletTransformConfiguration() {
        ShapeletTransform transform = new ShapeletTransform.Builder()
                .numShapelets(10)
                .minShapeletLength(3)
                .maxShapeletLength(5)
                .maxCandidates(100)
                .selectionMethod(ShapeletTransform.ShapeletSelectionMethod.INFORMATION_GAIN)
                .initializationMethod(ShapeletTransform.InitializationMethod.RANDOM)
                .removeSimilar(true)
                .similarityThreshold(0.1)
                .verbose(false)
                .randomSeed(42L)
                .build();
        
        assertNotNull(transform);
        assertFalse(transform.isFitted());
        assertEquals(0, transform.getNumShapelets());
    }
    
    @Test
    void testShapeletTransformFitTransform() {
        ShapeletTransform transform = new ShapeletTransform.Builder()
                .numShapelets(5)
                .minShapeletLength(3)
                .maxShapeletLength(4)
                .maxCandidates(50)
                .verbose(false)
                .randomSeed(42L)
                .build();
        
        // Treinar e transformar
        double[][] transformed = transform.fitTransform(testData, testLabels);
        
        assertTrue(transform.isFitted());
        assertEquals(testData.length, transformed.length);
        assertTrue(transform.getNumShapelets() > 0);
        assertTrue(transform.getNumShapelets() <= 5);
        
        // Verificar que shapelets foram descobertos
        assertNotNull(transform.getShapelets());
        assertFalse(transform.getShapelets().isEmpty());
        
        // Verificar classes
        String[] classes = transform.getClasses();
        assertNotNull(classes);
        assertTrue(classes.length >= 2);
    }
    
    @Test
    void testShapeletTransformPredict() {
        ShapeletTransform transform = new ShapeletTransform.Builder()
                .numShapelets(5)
                .minShapeletLength(3)
                .maxShapeletLength(4)
                .maxCandidates(50)
                .verbose(false)
                .randomSeed(42L)
                .build();
        
        // Treinar
        transform.fit(testData, testLabels);
        
        // Transformar novos dados
        double[][][] newData = Arrays.copyOf(testData, 5);
        double[][] transformed = transform.transform(newData);
        
        assertEquals(newData.length, transformed.length);
        assertEquals(transform.getNumShapelets(), transformed[0].length);
        
        // Verificar que as distâncias são não-negativas
        for (int i = 0; i < transformed.length; i++) {
            for (int j = 0; j < transformed[i].length; j++) {
                assertTrue(transformed[i][j] >= 0, 
                    "Distância deve ser não-negativa");
            }
        }
    }
    
    @Test
    void testShapeletTransformWithDifferentMethods() {
        ShapeletTransform.ShapeletSelectionMethod[] methods = {
            ShapeletTransform.ShapeletSelectionMethod.INFORMATION_GAIN,
            ShapeletTransform.ShapeletSelectionMethod.F_STATISTIC,
            ShapeletTransform.ShapeletSelectionMethod.MOODS_MEDIAN
        };
        
        for (ShapeletTransform.ShapeletSelectionMethod method : methods) {
            ShapeletTransform transform = new ShapeletTransform.Builder()
                    .numShapelets(3)
                    .selectionMethod(method)
                    .maxCandidates(30)
                    .verbose(false)
                    .randomSeed(42L)
                    .build();
            
            double[][] transformed = transform.fitTransform(testData, testLabels);
            
            assertTrue(transform.isFitted());
            assertEquals(testData.length, transformed.length);
            assertTrue(transform.getNumShapelets() > 0);
        }
    }
    
    @Test
    void testShapeletTransformWithDifferentInitialization() {
        ShapeletTransform.InitializationMethod[] methods = {
            ShapeletTransform.InitializationMethod.RANDOM,
            ShapeletTransform.InitializationMethod.CLASS_BALANCED
        };
        
        for (ShapeletTransform.InitializationMethod method : methods) {
            ShapeletTransform transform = new ShapeletTransform.Builder()
                    .numShapelets(3)
                    .initializationMethod(method)
                    .maxCandidates(30)
                    .verbose(false)
                    .randomSeed(42L)
                    .build();
            
            double[][] transformed = transform.fitTransform(testData, testLabels);
            
            assertTrue(transform.isFitted());
            assertEquals(testData.length, transformed.length);
        }
    }
    
    @Test
    void testErrorHandling() {
        ShapeletTransform transform = new ShapeletTransform.Builder().build();
        
        // Teste fit com dados inválidos
        assertThrows(IllegalArgumentException.class, () -> {
            transform.fit(null, testLabels);
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            transform.fit(testData, null);
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            double[][][] wrongData = new double[5][][];
            String[] wrongLabels = new String[10];
            transform.fit(wrongData, wrongLabels);
        });
        
        // Teste transform sem fit
        assertThrows(IllegalStateException.class, () -> {
            transform.transform(testData);
        });
    }
    
    @Test
    void testShapeletQualityEvaluation() {
        // Criar shapelet que deve ser discriminativo
        double[][] discriminativePattern = {{1.0}, {2.0}, {1.0}};
        Shapelet shapelet = new Shapelet(discriminativePattern, -1, -1, 0.0, "Test");
        
        // Criar dataset onde apenas uma classe tem este padrão
        double[][][] data = new double[10][10][1];
        String[] labels = new String[10];
        
        for (int i = 0; i < 10; i++) {
            labels[i] = (i < 5) ? "ClassA" : "ClassB";
            
            for (int t = 0; t < 10; t++) {
                data[i][t][0] = random.nextGaussian() * 0.1;
            }
            
            // Apenas ClassA tem o padrão discriminativo
            if (labels[i].equals("ClassA")) {
                data[i][0][0] = 1.0;
                data[i][1][0] = 2.0;
                data[i][2][0] = 1.0;
            }
        }
        
        // Treinar transform e verificar se encontra shapelets de boa qualidade
        ShapeletTransform transform = new ShapeletTransform.Builder()
                .numShapelets(5)
                .minShapeletLength(3)
                .maxShapeletLength(3)
                .maxCandidates(50)
                .verbose(false)
                .randomSeed(42L)
                .build();
        
        transform.fit(data, labels);
        
        // Deve encontrar pelo menos um shapelet com qualidade razoável
        double maxQuality = transform.getShapelets().stream()
                .mapToDouble(Shapelet::getQualityScore)
                .max().orElse(0.0);
        
        assertTrue(maxQuality > 0.5, "Deve encontrar shapelets discriminativos");
    }
}
