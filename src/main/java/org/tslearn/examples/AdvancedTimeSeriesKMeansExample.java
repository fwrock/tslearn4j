package org.tslearn.examples;

import java.util.Arrays;
import java.util.Random;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tslearn.clustering.TimeSeriesKMeans;

/**
 * Exemplo avançado demonstrando todas as funcionalidades do TimeSeriesKMeans.
 * 
 * Este exemplo mostra:
 * - Clustering com diferentes métricas
 * - Comparação de performance 
 * - Análise de qualidade dos clusters
 * - Predição de novos dados
 * - Visualização de resultados
 */
public class AdvancedTimeSeriesKMeansExample {
    
    private static final Logger logger = LoggerFactory.getLogger(AdvancedTimeSeriesKMeansExample.class);
    
    public static void main(String[] args) {
        logger.info("██████████████████████████████████████████████████████████████");
        logger.info("█████████ TimeSeriesKMeans - Exemplo Avançado ████████████████");
        logger.info("██████████████████████████████████████████████████████████████");
        logger.info("🚀 Clustering temporal com Java - equivalente ao Python tslearn");
        logger.info("██████████████████████████████████████████████████████████████");
        
        // 1. Gerar datasets com diferentes complexidades
        demonstrateDatasetGeneration();
        
        // 2. Clustering euclidiano vs DTW
        compareClusteringMethods();
        
        // 3. Análise de sensibilidade de parâmetros
        parameterSensitivityAnalysis();
        
        // 4. Clustering de séries temporais multivariadas
        multivariateTimeSeriesClustering();
        
        // 5. Pipeline completo de análise
        completeAnalysisPipeline();
        
        logger.info("██████████████████████████████████████████████████████████████");
        logger.info("✅ Exemplo avançado concluído com sucesso!");
        logger.info("🎯 TimeSeriesKMeans pronto para uso em produção");
        logger.info("██████████████████████████████████████████████████████████████");
    }
    
    /**
     * Demonstra a geração de diferentes tipos de datasets sintéticos.
     */
    private static void demonstrateDatasetGeneration() {
        logger.info("\n🔵 1. Geração de Datasets Sintéticos");
        logger.info("─────────────────────────────────────");
        
        // Dataset 1: Séries periódicas
        double[][][] periodicData = generatePeriodicTimeSeries(30, 60, 1);
        logger.info("📊 Dataset periódico: {} séries, {} timesteps", 
                   periodicData.length, periodicData[0].length);
        
        // Dataset 2: Séries com tendências
        double[][][] trendData = generateTrendTimeSeries(24, 80, 1);
        logger.info("📈 Dataset com tendências: {} séries, {} timesteps", 
                   trendData.length, trendData[0].length);
        
        // Dataset 3: Séries com ruído variável
        double[][][] noisyData = generateNoisyTimeSeries(36, 50, 2);
        logger.info("🔊 Dataset com ruído: {} séries, {} timesteps, {} features", 
                   noisyData.length, noisyData[0].length, noisyData[0][0].length);
    }
    
    /**
     * Compara métodos de clustering euclidiano vs DTW.
     */
    private static void compareClusteringMethods() {
        logger.info("\n🔴 2. Comparação Euclidiano vs DTW");
        logger.info("─────────────────────────────────────");
        
        // Gerar dataset com alinhamento temporal variável
        double[][][] dataset = generateVariableAlignmentDataset();
        
        // Teste Euclidiano
        long startTime = System.currentTimeMillis();
        TimeSeriesKMeans euclideanKMeans = new TimeSeriesKMeans.Builder()
                .nClusters(4)
                .metric(TimeSeriesKMeans.Metric.EUCLIDEAN)
                .maxIter(100)
                .nInit(10)
                .tolerance(1e-6)
                .randomSeed(42)
                .build();
        
        euclideanKMeans.fit(dataset);
        long euclideanTime = System.currentTimeMillis() - startTime;
        
        // Teste DTW
        startTime = System.currentTimeMillis();
        TimeSeriesKMeans dtwKMeans = new TimeSeriesKMeans.Builder()
                .nClusters(4)
                .metric(TimeSeriesKMeans.Metric.DTW)
                .maxIter(50)
                .maxIterBarycenter(15)
                .nInit(5)
                .tolerance(1e-4)
                .randomSeed(42)
                .build();
        
        dtwKMeans.fit(dataset);
        long dtwTime = System.currentTimeMillis() - startTime;
        
        // Análise comparativa
        logger.info("📊 Resultados da Comparação:");
        logger.info("   Euclidiano:");
        logger.info("     ⏱️  Tempo: {}ms", euclideanTime);
        logger.info("     🎯 Inércia: {:.6f}", euclideanKMeans.getInertia());
        logger.info("     🔄 Iterações: {}", euclideanKMeans.getNumIterations());
        
        logger.info("   DTW:");
        logger.info("     ⏱️  Tempo: {}ms", dtwTime);
        logger.info("     🎯 Inércia: {:.6f}", dtwKMeans.getInertia());
        logger.info("     🔄 Iterações: {}", dtwKMeans.getNumIterations());
        
        logger.info("   📈 Performance DTW/Euclidiano: {:.1f}x mais lento", 
                   (double) dtwTime / euclideanTime);
        
        // Comparar qualidade dos clusters
        compareClusterQuality(euclideanKMeans, dtwKMeans, dataset);
    }
    
    /**
     * Análise de sensibilidade de parâmetros.
     */
    private static void parameterSensitivityAnalysis() {
        logger.info("\n🟡 3. Análise de Sensibilidade de Parâmetros");
        logger.info("─────────────────────────────────────────────");
        
        double[][][] dataset = generatePeriodicTimeSeries(40, 60, 1);
        
        // Testar diferentes números de clusters
        int[] clusterCounts = {2, 3, 4, 5, 6};
        logger.info("🔢 Testando número de clusters:");
        
        for (int k : clusterCounts) {
            TimeSeriesKMeans kmeans = new TimeSeriesKMeans.Builder()
                    .nClusters(k)
                    .metric(TimeSeriesKMeans.Metric.EUCLIDEAN)
                    .maxIter(100)
                    .nInit(5)
                    .randomSeed(42)
                    .build();
            
            long startTime = System.currentTimeMillis();
            kmeans.fit(dataset);
            long elapsedTime = System.currentTimeMillis() - startTime;
            
            logger.info("   K={}: inércia={:.4f}, tempo={}ms", 
                       k, kmeans.getInertia(), elapsedTime);
        }
        
        // Testar diferentes inicializações
        logger.info("🎲 Testando número de inicializações:");
        int[] nInitValues = {1, 3, 5, 10, 20};
        
        for (int nInit : nInitValues) {
            TimeSeriesKMeans kmeans = new TimeSeriesKMeans.Builder()
                    .nClusters(4)
                    .metric(TimeSeriesKMeans.Metric.EUCLIDEAN)
                    .maxIter(100)
                    .nInit(nInit)
                    .randomSeed(42)
                    .build();
            
            long startTime = System.currentTimeMillis();
            kmeans.fit(dataset);
            long elapsedTime = System.currentTimeMillis() - startTime;
            
            logger.info("   n_init={}: inércia={:.4f}, tempo={}ms", 
                       nInit, kmeans.getInertia(), elapsedTime);
        }
    }
    
    /**
     * Clustering de séries temporais multivariadas.
     */
    private static void multivariateTimeSeriesClustering() {
        logger.info("\n🟢 4. Clustering Multivariado");
        logger.info("─────────────────────────────────");
        
        // Gerar dataset multivariado (3 features)
        double[][][] multivariateData = generateMultivariateTimeSeries(30, 40, 3);
        
        logger.info("📊 Dataset multivariado: {} séries, {} timesteps, {} features", 
                   multivariateData.length, multivariateData[0].length, multivariateData[0][0].length);
        
        // Clustering euclidiano
        TimeSeriesKMeans euclideanKMeans = new TimeSeriesKMeans.Builder()
                .nClusters(3)
                .metric(TimeSeriesKMeans.Metric.EUCLIDEAN)
                .maxIter(100)
                .nInit(5)
                .verbose(true)
                .randomSeed(42)
                .build();
        
        long startTime = System.currentTimeMillis();
        euclideanKMeans.fit(multivariateData);
        long euclideanTime = System.currentTimeMillis() - startTime;
        
        // Clustering DTW
        TimeSeriesKMeans dtwKMeans = new TimeSeriesKMeans.Builder()
                .nClusters(3)
                .metric(TimeSeriesKMeans.Metric.DTW)
                .maxIter(30)
                .maxIterBarycenter(10)
                .nInit(3)
                .verbose(true)
                .randomSeed(42)
                .build();
        
        startTime = System.currentTimeMillis();
        dtwKMeans.fit(multivariateData);
        long dtwTime = System.currentTimeMillis() - startTime;
        
        logger.info("📈 Resultados Multivariados:");
        logger.info("   Euclidiano: inércia={:.4f}, tempo={}ms", 
                   euclideanKMeans.getInertia(), euclideanTime);
        logger.info("   DTW:        inércia={:.4f}, tempo={}ms", 
                   dtwKMeans.getInertia(), dtwTime);
        
        // Analisar distribuição dos clusters
        analyzeClusterDistribution(euclideanKMeans.getLabels(), "Euclidiano");
        analyzeClusterDistribution(dtwKMeans.getLabels(), "DTW");
    }
    
    /**
     * Pipeline completo de análise.
     */
    private static void completeAnalysisPipeline() {
        logger.info("\n🟣 5. Pipeline Completo de Análise");
        logger.info("─────────────────────────────────────");
        
        // 1. Geração de dados
        logger.info("1️⃣  Gerando dataset de teste...");
        double[][][] dataset = generateComplexDataset(60, 80, 2);
        
        // 2. Clustering inicial
        logger.info("2️⃣  Executando clustering inicial...");
        TimeSeriesKMeans initialKMeans = new TimeSeriesKMeans.Builder()
                .nClusters(5)
                .metric(TimeSeriesKMeans.Metric.EUCLIDEAN)
                .maxIter(100)
                .nInit(10)
                .randomSeed(42)
                .build();
        
        long totalStartTime = System.currentTimeMillis();
        initialKMeans.fit(dataset);
        
        // 3. Refinamento com DTW
        logger.info("3️⃣  Refinando com DTW...");
        TimeSeriesKMeans refinedKMeans = new TimeSeriesKMeans.Builder()
                .nClusters(5)
                .metric(TimeSeriesKMeans.Metric.DTW)
                .maxIter(30)
                .maxIterBarycenter(15)
                .nInit(3)
                .randomSeed(42)
                .build();
        
        refinedKMeans.fit(dataset);
        
        // 4. Avaliação final
        logger.info("4️⃣  Avaliando resultados...");
        double[][] euclideanDistances = initialKMeans.transform(dataset);
        double[][] dtwDistances = refinedKMeans.transform(dataset);
        
        long totalTime = System.currentTimeMillis() - totalStartTime;
        
        // 5. Relatório final
        logger.info("📊 Relatório Final do Pipeline:");
        logger.info("   ⏱️  Tempo total: {}ms", totalTime);
        logger.info("   🎯 Inércia Euclidiana: {:.6f}", initialKMeans.getInertia());
        logger.info("   🎯 Inércia DTW: {:.6f}", refinedKMeans.getInertia());
        logger.info("   🔄 Total de iterações: {}", 
                   initialKMeans.getNumIterations() + refinedKMeans.getNumIterations());
        
        // Análise de estabilidade
        analyzeClusterStability(initialKMeans.getLabels(), refinedKMeans.getLabels());
        
        // Teste de predição
        logger.info("5️⃣  Testando capacidade de predição...");
        testPredictionCapability(refinedKMeans, dataset);
    }
    
    // Métodos auxiliares para geração de dados
    
    private static double[][][] generatePeriodicTimeSeries(int nSamples, int timeLength, int nFeatures) {
        double[][][] data = new double[nSamples][timeLength][nFeatures];
        Random random = new Random(42);
        
        for (int i = 0; i < nSamples; i++) {
            double frequency = 0.5 + (i % 3) * 0.5; // 3 frequências diferentes
            double phase = (i % 4) * Math.PI / 2;   // 4 fases diferentes
            
            for (int t = 0; t < timeLength; t++) {
                double time = (double) t / timeLength;
                for (int d = 0; d < nFeatures; d++) {
                    data[i][t][d] = Math.sin(2 * Math.PI * frequency * time + phase + d * Math.PI / 4) 
                                  + 0.1 * random.nextGaussian();
                }
            }
        }
        
        return data;
    }
    
    private static double[][][] generateTrendTimeSeries(int nSamples, int timeLength, int nFeatures) {
        double[][][] data = new double[nSamples][timeLength][nFeatures];
        Random random = new Random(123);
        
        for (int i = 0; i < nSamples; i++) {
            double slope = -2.0 + (i % 5) * 1.0;  // 5 inclinações diferentes
            double intercept = random.nextGaussian();
            
            for (int t = 0; t < timeLength; t++) {
                double time = (double) t / timeLength;
                for (int d = 0; d < nFeatures; d++) {
                    data[i][t][d] = slope * time + intercept + 0.2 * random.nextGaussian();
                }
            }
        }
        
        return data;
    }
    
    private static double[][][] generateNoisyTimeSeries(int nSamples, int timeLength, int nFeatures) {
        double[][][] data = new double[nSamples][timeLength][nFeatures];
        Random random = new Random(456);
        
        for (int i = 0; i < nSamples; i++) {
            double noiseLevel = 0.1 + (i % 4) * 0.3;  // 4 níveis de ruído
            
            for (int t = 0; t < timeLength; t++) {
                for (int d = 0; d < nFeatures; d++) {
                    data[i][t][d] = noiseLevel * random.nextGaussian();
                }
            }
        }
        
        return data;
    }
    
    private static double[][][] generateVariableAlignmentDataset() {
        int nSamples = 40;
        int timeLength = 60;
        double[][][] data = new double[nSamples][timeLength][1];
        Random random = new Random(789);
        
        for (int i = 0; i < nSamples; i++) {
            int pattern = i % 4;
            int shift = random.nextInt(10) - 5; // Deslocamento temporal
            
            for (int t = 0; t < timeLength; t++) {
                double adjustedTime = (double) (t + shift) / timeLength;
                switch (pattern) {
                    case 0: // Pico no início
                        data[i][t][0] = Math.exp(-5 * adjustedTime) + 0.1 * random.nextGaussian();
                        break;
                    case 1: // Pico no meio
                        data[i][t][0] = Math.exp(-5 * Math.abs(adjustedTime - 0.5)) + 0.1 * random.nextGaussian();
                        break;
                    case 2: // Pico no final
                        data[i][t][0] = Math.exp(-5 * (1 - adjustedTime)) + 0.1 * random.nextGaussian();
                        break;
                    case 3: // Dois picos
                        data[i][t][0] = Math.exp(-10 * Math.abs(adjustedTime - 0.3)) + 
                                       Math.exp(-10 * Math.abs(adjustedTime - 0.7)) + 
                                       0.1 * random.nextGaussian();
                        break;
                }
            }
        }
        
        return data;
    }
    
    private static double[][][] generateMultivariateTimeSeries(int nSamples, int timeLength, int nFeatures) {
        double[][][] data = new double[nSamples][timeLength][nFeatures];
        Random random = new Random(101112);
        
        for (int i = 0; i < nSamples; i++) {
            int cluster = i % 3; // 3 clusters
            
            for (int t = 0; t < timeLength; t++) {
                double time = (double) t / timeLength;
                
                switch (cluster) {
                    case 0: // Padrão sinusoidal correlacionado
                        for (int d = 0; d < nFeatures; d++) {
                            data[i][t][d] = Math.sin(2 * Math.PI * time + d * Math.PI / 6) + 
                                          0.1 * random.nextGaussian();
                        }
                        break;
                    case 1: // Padrão linear com diferentes inclinações
                        for (int d = 0; d < nFeatures; d++) {
                            data[i][t][d] = (d + 1) * time + 0.1 * random.nextGaussian();
                        }
                        break;
                    case 2: // Padrão exponencial
                        for (int d = 0; d < nFeatures; d++) {
                            data[i][t][d] = Math.exp(-(d + 1) * time) + 0.1 * random.nextGaussian();
                        }
                        break;
                }
            }
        }
        
        return data;
    }
    
    private static double[][][] generateComplexDataset(int nSamples, int timeLength, int nFeatures) {
        // Combinar diferentes tipos de séries temporais
        double[][][] periodic = generatePeriodicTimeSeries(nSamples / 3, timeLength, nFeatures);
        double[][][] trend = generateTrendTimeSeries(nSamples / 3, timeLength, nFeatures);
        double[][][] noisy = generateNoisyTimeSeries(nSamples - 2 * (nSamples / 3), timeLength, nFeatures);
        
        double[][][] combined = new double[nSamples][timeLength][nFeatures];
        System.arraycopy(periodic, 0, combined, 0, periodic.length);
        System.arraycopy(trend, 0, combined, periodic.length, trend.length);
        System.arraycopy(noisy, 0, combined, periodic.length + trend.length, noisy.length);
        
        return combined;
    }
    
    // Métodos auxiliares para análise
    
    private static void compareClusterQuality(TimeSeriesKMeans kmeans1, TimeSeriesKMeans kmeans2, 
                                            double[][][] dataset) {
        logger.info("🔍 Análise de Qualidade dos Clusters:");
        
        // Calcular silhouette score simplificado
        double silhouette1 = calculateSimplifiedSilhouette(kmeans1, dataset);
        double silhouette2 = calculateSimplifiedSilhouette(kmeans2, dataset);
        
        logger.info("   Silhouette Euclidiano: {:.4f}", silhouette1);
        logger.info("   Silhouette DTW:        {:.4f}", silhouette2);
        
        if (silhouette2 > silhouette1) {
            logger.info("   🏆 DTW produz clusters de melhor qualidade");
        } else {
            logger.info("   🏆 Euclidiano produz clusters de melhor qualidade");
        }
    }
    
    private static double calculateSimplifiedSilhouette(TimeSeriesKMeans kmeans, double[][][] dataset) {
        // Implementação simplificada do silhouette score
        int[] labels = kmeans.getLabels();
        double[][] distances = kmeans.transform(dataset);
        
        double totalSilhouette = 0.0;
        int nSamples = dataset.length;
        
        for (int i = 0; i < nSamples; i++) {
            double intraClusterDist = distances[i][labels[i]];
            
            double minInterClusterDist = Double.POSITIVE_INFINITY;
            for (int k = 0; k < kmeans.getNumClusters(); k++) {
                if (k != labels[i]) {
                    minInterClusterDist = Math.min(minInterClusterDist, distances[i][k]);
                }
            }
            
            if (Math.max(intraClusterDist, minInterClusterDist) > 0) {
                double silhouette = (minInterClusterDist - intraClusterDist) / 
                                  Math.max(intraClusterDist, minInterClusterDist);
                totalSilhouette += silhouette;
            }
        }
        
        return totalSilhouette / nSamples;
    }
    
    private static void analyzeClusterDistribution(int[] labels, String method) {
        int[] counts = new int[Arrays.stream(labels).max().orElse(0) + 1];
        for (int label : labels) {
            counts[label]++;
        }
        
        logger.info("   Distribuição {}: {}", method, Arrays.toString(counts));
    }
    
    private static void analyzeClusterStability(int[] labels1, int[] labels2) {
        int agreement = 0;
        for (int i = 0; i < labels1.length; i++) {
            if (labels1[i] == labels2[i]) {
                agreement++;
            }
        }
        
        double stability = (double) agreement / labels1.length;
        logger.info("🔄 Estabilidade dos clusters: {:.1f}% ({}/{})", 
                   stability * 100, agreement, labels1.length);
    }
    
    private static void testPredictionCapability(TimeSeriesKMeans kmeans, double[][][] dataset) {
        // Usar 20% dos dados para teste
        int testSize = dataset.length / 5;
        double[][][] testData = Arrays.copyOf(dataset, testSize);
        
        int[] predictions = kmeans.predict(testData);
        double[][] distances = kmeans.transform(testData);
        
        logger.info("   📊 Testado em {} amostras", testSize);
        logger.info("   🎯 Predições: {}", Arrays.toString(predictions));
        
        // Calcular confiança média
        double avgConfidence = 0.0;
        for (int i = 0; i < testSize; i++) {
            double minDist = Arrays.stream(distances[i]).min().orElse(0.0);
            double maxDist = Arrays.stream(distances[i]).max().orElse(1.0);
            avgConfidence += (maxDist - minDist) / maxDist;
        }
        avgConfidence /= testSize;
        
        logger.info("   🎓 Confiança média: {:.1f}%", avgConfidence * 100);
    }
}
