package org.tslearn.early_classification;

import java.util.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/**
 * Testes unitários para Early Classification.
 */
public class EarlyClassificationTest {
    
    private double[][][] trainingData;
    private String[] trainingLabels;
    private double[][][] testData;
    private String[] testLabels;
    
    @BeforeEach
    void setUp() {
        // Criar dataset sintético simples
        Random random = new Random(42);
        
        // Dados de treinamento
        trainingData = new double[20][10][1];
        trainingLabels = new String[20];
        
        for (int i = 0; i < 20; i++) {
            String label = (i < 10) ? "ClasseA" : "ClasseB";
            trainingLabels[i] = label;
            
            for (int t = 0; t < 10; t++) {
                if (label.equals("ClasseA")) {
                    // Padrão crescente
                    trainingData[i][t][0] = t * 0.5 + random.nextGaussian() * 0.1;
                } else {
                    // Padrão decrescente
                    trainingData[i][t][0] = (10 - t) * 0.5 + random.nextGaussian() * 0.1;
                }
            }
        }
        
        // Dados de teste
        testData = new double[10][10][1];
        testLabels = new String[10];
        
        for (int i = 0; i < 10; i++) {
            String label = (i < 5) ? "ClasseA" : "ClasseB";
            testLabels[i] = label;
            
            for (int t = 0; t < 10; t++) {
                if (label.equals("ClasseA")) {
                    testData[i][t][0] = t * 0.5 + random.nextGaussian() * 0.1;
                } else {
                    testData[i][t][0] = (10 - t) * 0.5 + random.nextGaussian() * 0.1;
                }
            }
        }
    }
    
    @Test
    void testEarlyClassifierBasicFunctionality() {
        EarlyClassifier classifier = new EarlyClassifier();
        
        // Verificar estado inicial
        assertFalse(classifier.isFitted());
        
        // Treinar
        assertDoesNotThrow(() -> classifier.fit(trainingData, trainingLabels));
        assertTrue(classifier.isFitted());
        
        // Verificar classes
        String[] classes = classifier.getClasses();
        assertNotNull(classes);
        assertEquals(2, classes.length);
        assertTrue(Arrays.asList(classes).contains("ClasseA"));
        assertTrue(Arrays.asList(classes).contains("ClasseB"));
    }
    
    @Test
    void testEarlyClassifierBuilder() {
        EarlyClassifier classifier = new EarlyClassifier.Builder()
            .stoppingStrategy(EarlyClassifier.StoppingStrategy.CONFIDENCE_THRESHOLD)
            .confidenceThreshold(0.8)
            .minLength(3)
            .stepSize(1)
            .verbose(false)
            .build();
        
        assertEquals(EarlyClassifier.StoppingStrategy.CONFIDENCE_THRESHOLD, classifier.getStoppingStrategy());
        assertEquals(0.8, classifier.getConfidenceThreshold(), 1e-6);
        assertFalse(classifier.isFitted());
    }
    
    @Test
    void testEarlyPrediction() {
        EarlyClassifier classifier = new EarlyClassifier.Builder()
            .confidenceThreshold(0.7)
            .minLength(3)
            .stepSize(1)
            .build();
        
        classifier.fit(trainingData, trainingLabels);
        
        // Predição early em uma série de teste
        EarlyClassifier.EarlyResult result = classifier.predictEarly(testData[0]);
        
        assertNotNull(result);
        assertNotNull(result.getPredictedClass());
        assertTrue(result.getConfidence() >= 0.0 && result.getConfidence() <= 1.0);
        assertTrue(result.getStoppingTime() >= 3);
        assertTrue(result.getEarliness() >= 0.0 && result.getEarliness() <= 1.0);
        assertNotNull(result.getClassProbabilities());
        assertNotNull(result.getSteps());
        assertFalse(result.getSteps().isEmpty());
    }
    
    @Test
    void testMultiplePredictions() {
        EarlyClassifier classifier = new EarlyClassifier.Builder()
            .confidenceThreshold(0.8)
            .minLength(2)
            .build();
        
        classifier.fit(trainingData, trainingLabels);
        
        List<EarlyClassifier.EarlyResult> results = classifier.predictEarly(testData);
        
        assertNotNull(results);
        assertEquals(testData.length, results.size());
        
        for (EarlyClassifier.EarlyResult result : results) {
            assertNotNull(result.getPredictedClass());
            assertTrue(result.getConfidence() >= 0.0);
            assertTrue(result.getEarliness() >= 0.0);
        }
    }
    
    @Test
    void testEvaluation() {
        EarlyClassifier classifier = new EarlyClassifier.Builder()
            .confidenceThreshold(0.7)
            .minLength(3)
            .build();
        
        classifier.fit(trainingData, trainingLabels);
        
        EarlyClassifier.EvaluationResult evaluation = classifier.evaluate(testData, testLabels);
        
        assertNotNull(evaluation);
        assertTrue(evaluation.getAccuracy() >= 0.0 && evaluation.getAccuracy() <= 1.0);
        assertTrue(evaluation.getAverageEarliness() >= 0.0 && evaluation.getAverageEarliness() <= 1.0);
        assertTrue(evaluation.getAverageConfidence() >= 0.0 && evaluation.getAverageConfidence() <= 1.0);
        assertTrue(evaluation.getHarmonicMean() >= 0.0);
        assertEquals(testData.length, evaluation.getResults().size());
    }
    
    @Test
    void testDifferentStoppingStrategies() {
        EarlyClassifier.StoppingStrategy[] strategies = {
            EarlyClassifier.StoppingStrategy.CONFIDENCE_THRESHOLD,
            EarlyClassifier.StoppingStrategy.MARGIN_BASED,
            EarlyClassifier.StoppingStrategy.PROBABILITY_STABILIZATION
        };
        
        for (EarlyClassifier.StoppingStrategy strategy : strategies) {
            EarlyClassifier classifier = new EarlyClassifier.Builder()
                .stoppingStrategy(strategy)
                .confidenceThreshold(0.8)
                .marginThreshold(0.3)
                .minLength(2)
                .build();
            
            assertDoesNotThrow(() -> classifier.fit(trainingData, trainingLabels));
            assertTrue(classifier.isFitted());
            
            EarlyClassifier.EarlyResult result = classifier.predictEarly(testData[0]);
            assertNotNull(result);
            assertNotNull(result.getPredictedClass());
        }
    }
    
    @Test
    void testDifferentAggregationMethods() {
        EarlyClassifier.AggregationMethod[] methods = {
            EarlyClassifier.AggregationMethod.MAJORITY_VOTE,
            EarlyClassifier.AggregationMethod.PROBABILITY_AVERAGE,
            EarlyClassifier.AggregationMethod.WEIGHTED_CONFIDENCE,
            EarlyClassifier.AggregationMethod.MAX_CONFIDENCE
        };
        
        for (EarlyClassifier.AggregationMethod method : methods) {
            EarlyClassifier classifier = new EarlyClassifier.Builder()
                .aggregationMethod(method)
                .confidenceThreshold(0.7)
                .minLength(2)
                .build();
            
            assertDoesNotThrow(() -> classifier.fit(trainingData, trainingLabels));
            
            EarlyClassifier.EarlyResult result = classifier.predictEarly(testData[0]);
            assertNotNull(result);
        }
    }
    
    @Test
    void testCustomClassifiers() {
        EarlyClassifier classifier = new EarlyClassifier.Builder()
            .addClassifier(new DTWNearestNeighbor())
            .addClassifier(new FeatureBasedClassifier())
            .confidenceThreshold(0.8)
            .minLength(3)
            .build();
        
        assertEquals(2, classifier.getNumClassifiers());
        
        assertDoesNotThrow(() -> classifier.fit(trainingData, trainingLabels));
        assertTrue(classifier.isFitted());
        
        EarlyClassifier.EarlyResult result = classifier.predictEarly(testData[0]);
        assertNotNull(result);
    }
    
    @Test
    void testErrorHandling() {
        EarlyClassifier classifier = new EarlyClassifier();
        
        // Predição sem treinamento
        assertThrows(RuntimeException.class, () -> classifier.predictEarly(testData[0]));
        
        // Dados null
        assertThrows(IllegalArgumentException.class, () -> classifier.fit(null, trainingLabels));
        assertThrows(IllegalArgumentException.class, () -> classifier.fit(trainingData, null));
        
        // Dimensões incompatíveis
        String[] wrongLabels = new String[5];
        assertThrows(IllegalArgumentException.class, () -> classifier.fit(trainingData, wrongLabels));
        
        // Dataset vazio
        assertThrows(IllegalArgumentException.class, () -> classifier.fit(new double[0][][], new String[0]));
        
        // Treinar e tentar predição com série null/vazia
        classifier.fit(trainingData, trainingLabels);
        assertThrows(IllegalArgumentException.class, () -> classifier.predictEarly((double[][])null));
        assertThrows(IllegalArgumentException.class, () -> classifier.predictEarly(new double[0][]));
    }
    
    @Test
    void testBuilderValidation() {
        EarlyClassifier.Builder builder = new EarlyClassifier.Builder();
        
        // Threshold inválido
        assertThrows(IllegalArgumentException.class, () -> builder.confidenceThreshold(0.0));
        assertThrows(IllegalArgumentException.class, () -> builder.confidenceThreshold(1.5));
        assertThrows(IllegalArgumentException.class, () -> builder.marginThreshold(-0.1));
        assertThrows(IllegalArgumentException.class, () -> builder.marginThreshold(1.5));
        
        // Comprimentos inválidos
        assertThrows(IllegalArgumentException.class, () -> builder.minLength(0));
        assertThrows(IllegalArgumentException.class, () -> builder.stepSize(0));
        
        // Valores válidos
        assertDoesNotThrow(() -> builder.confidenceThreshold(0.8));
        assertDoesNotThrow(() -> builder.marginThreshold(0.3));
        assertDoesNotThrow(() -> builder.minLength(5));
        assertDoesNotThrow(() -> builder.stepSize(2));
    }
    
    @Test
    void testClassificationSteps() {
        EarlyClassifier classifier = new EarlyClassifier.Builder()
            .confidenceThreshold(0.9) // Alto para forçar mais passos
            .minLength(2)
            .stepSize(1)
            .build();
        
        classifier.fit(trainingData, trainingLabels);
        EarlyClassifier.EarlyResult result = classifier.predictEarly(testData[0]);
        
        List<EarlyClassifier.ClassificationStep> steps = result.getSteps();
        assertNotNull(steps);
        assertFalse(steps.isEmpty());
        
        // Verificar consistência dos passos
        for (EarlyClassifier.ClassificationStep step : steps) {
            assertTrue(step.getTimePoint() >= 2); // Min length
            assertNotNull(step.getProbabilities());
            assertTrue(step.getConfidence() >= 0.0);
            
            // Probabilidades devem somar aproximadamente 1
            double sum = step.getProbabilities().values().stream().mapToDouble(Double::doubleValue).sum();
            assertEquals(1.0, sum, 0.1);
        }
    }
    
    @Test
    void testPerformanceConsistency() {
        EarlyClassifier classifier1 = new EarlyClassifier.Builder()
            .randomSeed(42)
            .confidenceThreshold(0.8)
            .build();
        
        EarlyClassifier classifier2 = new EarlyClassifier.Builder()
            .randomSeed(42)
            .confidenceThreshold(0.8)
            .build();
        
        // Treinar ambos com mesmos dados
        classifier1.fit(trainingData, trainingLabels);
        classifier2.fit(trainingData, trainingLabels);
        
        // Resultados devem ser consistentes (mesma seed)
        EarlyClassifier.EarlyResult result1 = classifier1.predictEarly(testData[0]);
        EarlyClassifier.EarlyResult result2 = classifier2.predictEarly(testData[0]);
        
        assertEquals(result1.getPredictedClass(), result2.getPredictedClass());
        assertEquals(result1.getStoppingTime(), result2.getStoppingTime());
    }
}
