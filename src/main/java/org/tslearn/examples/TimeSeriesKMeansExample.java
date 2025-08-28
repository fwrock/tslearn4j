package org.tslearn.examples;

import java.util.Arrays;
import java.util.Random;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tslearn.clustering.TimeSeriesKMeans;

/**
 * Exemplo demonstrando o uso do TimeSeriesKMeans com diferentes métricas.
 */
public class TimeSeriesKMeansExample {
    
    private static final Logger logger = LoggerFactory.getLogger(TimeSeriesKMeansExample.class);
    
    public static void main(String[] args) {
        logger.info("🚀 Demonstração TimeSeriesKMeans");
        logger.info("═════════════════════════════════");
        
        // Gerar dataset sintético
        double[][][] dataset = generateSyntheticDataset();
        logger.info("📊 Dataset: {} séries temporais, {} timesteps, {} features", 
                   dataset.length, dataset[0].length, dataset[0][0].length);
        
        // Teste 1: K-means Euclidiano
        testEuclideanKMeans(dataset);
        
        // Teste 2: K-means com DTW
        testDTWKMeans(dataset);
        
        // Teste 3: Comparação de performance
        comparePerformance(dataset);
        
        logger.info("✅ Demonstração concluída!");
    }
    
    /**
     * Gera um dataset sintético com 3 padrões distintos.
     */
    private static double[][][] generateSyntheticDataset() {
        int nSamples = 30;
        int timeLength = 50;
        int nFeatures = 2;
        
        double[][][] dataset = new double[nSamples][timeLength][nFeatures];
        Random random = new Random(42);
        
        for (int i = 0; i < nSamples; i++) {
            int pattern = i % 3; // 3 padrões diferentes
            
            for (int t = 0; t < timeLength; t++) {
                double time = (double) t / timeLength;
                
                switch (pattern) {
                    case 0: // Padrão senoidal
                        dataset[i][t][0] = Math.sin(2 * Math.PI * time) + 0.1 * random.nextGaussian();
                        dataset[i][t][1] = Math.cos(2 * Math.PI * time) + 0.1 * random.nextGaussian();
                        break;
                    case 1: // Padrão linear crescente
                        dataset[i][t][0] = 2 * time - 1 + 0.1 * random.nextGaussian();
                        dataset[i][t][1] = time + 0.1 * random.nextGaussian();
                        break;
                    case 2: // Padrão exponencial
                        dataset[i][t][0] = Math.exp(-time) + 0.1 * random.nextGaussian();
                        dataset[i][t][1] = Math.exp(-2 * time) + 0.1 * random.nextGaussian();
                        break;
                }
            }
        }
        
        return dataset;
    }
    
    /**
     * Testa K-means com métrica euclidiana.
     */
    private static void testEuclideanKMeans(double[][][] dataset) {
        logger.info("\n🔵 Teste 1: K-means Euclidiano");
        logger.info("─────────────────────────────────");
        
        long startTime = System.currentTimeMillis();
        
        TimeSeriesKMeans kmeans = new TimeSeriesKMeans.Builder()
                .nClusters(3)
                .metric(TimeSeriesKMeans.Metric.EUCLIDEAN)
                .maxIter(100)
                .nInit(5)
                .tolerance(1e-6)
                .verbose(true)
                .randomSeed(42)
                .build();
        
        kmeans.fit(dataset);
        
        long elapsedTime = System.currentTimeMillis() - startTime;
        
        // Resultados
        int[] labels = kmeans.getLabels();
        double inertia = kmeans.getInertia();
        int nIter = kmeans.getNumIterations();
        
        logger.info("⏱️  Tempo de execução: {}ms", elapsedTime);
        logger.info("🔄 Convergiu em: {} iterações", nIter);
        logger.info("🎯 Inércia final: {:.6f}", inertia);
        
        // Distribuição dos clusters
        int[] clusterCounts = new int[3];
        for (int label : labels) {
            clusterCounts[label]++;
        }
        logger.info("📊 Distribuição: C0={} C1={} C2={}", 
                   clusterCounts[0], clusterCounts[1], clusterCounts[2]);
    }
    
    /**
     * Testa K-means com DTW.
     */
    private static void testDTWKMeans(double[][][] dataset) {
        logger.info("\n🔴 Teste 2: K-means com DTW");
        logger.info("─────────────────────────────────");
        
        long startTime = System.currentTimeMillis();
        
        TimeSeriesKMeans kmeans = new TimeSeriesKMeans.Builder()
                .nClusters(3)
                .metric(TimeSeriesKMeans.Metric.DTW)
                .maxIter(50)
                .maxIterBarycenter(10)
                .nInit(3)
                .tolerance(1e-4)
                .verbose(true)
                .randomSeed(42)
                .build();
        
        kmeans.fit(dataset);
        
        long elapsedTime = System.currentTimeMillis() - startTime;
        
        // Resultados
        int[] labels = kmeans.getLabels();
        double inertia = kmeans.getInertia();
        int nIter = kmeans.getNumIterations();
        
        logger.info("⏱️  Tempo de execução: {}ms", elapsedTime);
        logger.info("🔄 Convergiu em: {} iterações", nIter);
        logger.info("🎯 Inércia final: {:.6f}", inertia);
        
        // Distribuição dos clusters
        int[] clusterCounts = new int[3];
        for (int label : labels) {
            clusterCounts[label]++;
        }
        logger.info("📊 Distribuição: C0={} C1={} C2={}", 
                   clusterCounts[0], clusterCounts[1], clusterCounts[2]);
    }
    
    /**
     * Compara performance entre métricas.
     */
    private static void comparePerformance(double[][][] dataset) {
        logger.info("\n🟡 Teste 3: Comparação de Performance");
        logger.info("─────────────────────────────────────");
        
        // Teste Euclidiano
        long startTime = System.currentTimeMillis();
        TimeSeriesKMeans euclideanKMeans = new TimeSeriesKMeans.Builder()
                .nClusters(3)
                .metric(TimeSeriesKMeans.Metric.EUCLIDEAN)
                .maxIter(100)
                .nInit(3)
                .randomSeed(42)
                .build();
        euclideanKMeans.fit(dataset);
        long euclideanTime = System.currentTimeMillis() - startTime;
        
        // Teste DTW
        startTime = System.currentTimeMillis();
        TimeSeriesKMeans dtwKMeans = new TimeSeriesKMeans.Builder()
                .nClusters(3)
                .metric(TimeSeriesKMeans.Metric.DTW)
                .maxIter(50)
                .maxIterBarycenter(5)
                .nInit(3)
                .randomSeed(42)
                .build();
        dtwKMeans.fit(dataset);
        long dtwTime = System.currentTimeMillis() - startTime;
        
        // Comparação
        logger.info("📈 Resultados da Comparação:");
        logger.info("   Euclidiano: {:.6f} inércia, {}ms", 
                   euclideanKMeans.getInertia(), euclideanTime);
        logger.info("   DTW:        {:.6f} inércia, {}ms", 
                   dtwKMeans.getInertia(), dtwTime);
        logger.info("   Speedup DTW/Euclidiano: {:.1f}x", 
                   (double) dtwTime / euclideanTime);
        
        // Teste de predição
        testPrediction(euclideanKMeans, dtwKMeans, dataset);
    }
    
    /**
     * Testa a capacidade de predição dos modelos.
     */
    private static void testPrediction(TimeSeriesKMeans euclideanModel, 
                                      TimeSeriesKMeans dtwModel, 
                                      double[][][] testData) {
        logger.info("\n🔍 Teste de Predição:");
        
        // Usar as primeiras 5 amostras para teste
        double[][][] testSamples = Arrays.copyOf(testData, 5);
        
        int[] euclideanPreds = euclideanModel.predict(testSamples);
        int[] dtwPreds = dtwModel.predict(testSamples);
        
        logger.info("   Predições Euclidiano: {}", Arrays.toString(euclideanPreds));
        logger.info("   Predições DTW:        {}", Arrays.toString(dtwPreds));
        
        // Calcular concordância
        int agreement = 0;
        for (int i = 0; i < euclideanPreds.length; i++) {
            if (euclideanPreds[i] == dtwPreds[i]) {
                agreement++;
            }
        }
        logger.info("   Concordância: {}/{} ({:.1f}%)", 
                   agreement, euclideanPreds.length, 
                   100.0 * agreement / euclideanPreds.length);
    }
}
