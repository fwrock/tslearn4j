package org.tslearn.examples;

import java.util.List;
import java.util.Random;
import org.apache.commons.math3.linear.RealMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tslearn.clustering.KShape;
import org.tslearn.metrics.DTW;
import org.tslearn.metrics.DTWNeighbors;
import org.tslearn.preprocessing.TimeSeriesScalerMeanVariance;
import org.tslearn.utils.MatrixUtils;

/**
 * Demonstração completa da biblioteca TSLearn4J.
 * 
 * Este exemplo mostra todas as funcionalidades implementadas:
 * - KShape clustering otimizado com FFT
 * - DTW otimizada com múltiplas estratégias de aceleração
 * - Preprocessamento de dados
 * - Busca de vizinhos mais próximos
 */
public class TSLearn4JShowcase {
    
    private static final Logger logger = LoggerFactory.getLogger(TSLearn4JShowcase.class);
    
    public static void main(String[] args) {
        logger.info("█".repeat(80));
        logger.info("██████████████████ TSLearn4J Complete Showcase ██████████████████");
        logger.info("█".repeat(80));
        logger.info("🚀 Biblioteca Java otimizada para Time Series Machine Learning");
        logger.info("📊 Equivalente ao Python tslearn com otimizações de performance");
        logger.info("█".repeat(80));
        
        // 1. Clustering com KShape
        demonstrateKShapeClustering();
        
        // 2. DTW e métricas de distância
        demonstrateDTWMetrics();
        
        // 3. Busca de vizinhos mais próximos
        demonstrateNearestNeighbors();
        
        // 4. Pipeline completo de análise
        demonstrateCompletePipeline();
        
        logger.info("\n" + "█".repeat(80));
        logger.info("✅ TSLearn4J: Implementação completa e otimizada!");
        logger.info("🎯 Pronta para uso em produção com performance superior");
        logger.info("█".repeat(80));
    }
    
    private static void demonstrateKShapeClustering() {
        logger.info("\n🔵 1. KShape Clustering com FFT Otimizada");
        logger.info("─".repeat(60));
        
        // Gerar dados sintéticos com padrões distintos
        Random random = new Random(42);
        double[][] data = new double[20][];
        
        // Cluster 1: Ondas seno
        for (int i = 0; i < 7; i++) {
            data[i] = generateSineWave(50, 1.0 + random.nextGaussian() * 0.1, 
                                      random.nextGaussian() * 0.5, 1.0);
        }
        
        // Cluster 2: Ondas cosseno
        for (int i = 7; i < 14; i++) {
            data[i] = generateCosineWave(50, 1.0 + random.nextGaussian() * 0.1, 
                                        random.nextGaussian() * 0.5, 1.0);
        }
        
        // Cluster 3: Random walks
        for (int i = 14; i < 20; i++) {
            data[i] = generateRandomWalk(50, random);
        }
        
        // Normalizar dados
        TimeSeriesScalerMeanVariance scaler = new TimeSeriesScalerMeanVariance();
        RealMatrix[] normalizedData = scaler.fitTransform(MatrixUtils.toTimeSeriesDataset(data));
        
        // KShape clustering
        KShape kshape = new KShape(3, 100, 1e-4, 5, false, 42L, "random");
        
        long startTime = System.currentTimeMillis();
        int[] labels = kshape.fitPredict(normalizedData);
        long endTime = System.currentTimeMillis();
        
        logger.info("📈 Dataset: {} séries temporais de {} pontos cada", data.length, data[0].length);
        logger.info("⚙️  Parâmetros: 3 clusters, 100 max_iter, 5 n_init");
        logger.info("⏱️  Tempo de execução: {}ms", endTime - startTime);
        logger.info("🔄 Convergiu em: {} iterações", kshape.getNIter());
        logger.info("🎯 Inércia final: {:.6f}", kshape.getInertia());
        
        // Analisar resultados
        int[] clusterCounts = new int[3];
        for (int label : labels) {
            if (label >= 0 && label < 3) clusterCounts[label]++;
        }
        
        logger.info("📊 Distribuição: C0={} C1={} C2={}", 
                   clusterCounts[0], clusterCounts[1], clusterCounts[2]);
        
        // FFT foi usada?
        boolean usingFFT = data[0].length > 64;
        logger.info("⚡ FFT optimization: {}", 
                   usingFFT ? "ATIVADA (série longa)" : "DESATIVADA (série curta)");
    }
    
    private static void demonstrateDTWMetrics() {
        logger.info("\n🔴 2. DTW Otimizada e Métricas de Distância");
        logger.info("─".repeat(60));
        
        double[] ts1 = generateSineWave(40, 1.0, 0.0, 1.0);
        double[] ts2 = generateSineWave(40, 1.1, Math.PI/4, 1.2);
        double[] ts3 = generateRandomWalk(40, new Random(123));
        
        // DTW padrão vs restrições
        DTW standardDTW = new DTW();
        DTW sakoeChibaDTW = new DTW(8);
        DTW itakuraDTW = new DTW(DTW.GlobalConstraint.ITAKURA, 0.0, false, Double.POSITIVE_INFINITY);
        
        long startTime = System.nanoTime();
        double dist1 = standardDTW.distance(ts1, ts2);
        long time1 = System.nanoTime() - startTime;
        
        startTime = System.nanoTime();
        double dist2 = sakoeChibaDTW.distance(ts1, ts2);
        long time2 = System.nanoTime() - startTime;
        
        startTime = System.nanoTime();
        double dist3 = itakuraDTW.distance(ts1, ts2);
        long time3 = System.nanoTime() - startTime;
        
        logger.info("📏 Distâncias DTW (séries de {} pontos):", ts1.length);
        logger.info("   Standard:    {:.6f} ({}μs)", dist1, time1 / 1000);
        logger.info("   Sakoe-Chiba: {:.6f} ({}μs) - Speedup: {:.1f}x", 
                   dist2, time2 / 1000, (double)time1 / time2);
        logger.info("   Itakura:     {:.6f} ({}μs) - Speedup: {:.1f}x", 
                   dist3, time3 / 1000, (double)time1 / time3);
        
        // Path alignment
        DTW.DTWResult result = standardDTW.distanceWithPath(ts1, ts2);
        logger.info("🛤️  Path alignment: distância={:.6f}, comprimento={}", 
                   result.getDistance(), result.getPathLength());
    }
    
    private static void demonstrateNearestNeighbors() {
        logger.info("\n🟢 3. Busca de Vizinhos Mais Próximos");
        logger.info("─".repeat(60));
        
        // Criar dataset
        Random random = new Random(456);
        int datasetSize = 100;
        int seriesLength = 35;
        
        double[][] dataset = new double[datasetSize][];
        for (int i = 0; i < datasetSize; i++) {
            if (i < 30) {
                dataset[i] = generateSineWave(seriesLength, 1.0 + random.nextGaussian() * 0.2, 
                                             random.nextDouble() * 2 * Math.PI, 1.0);
            } else if (i < 60) {
                dataset[i] = generateCosineWave(seriesLength, 1.0 + random.nextGaussian() * 0.2, 
                                               random.nextDouble() * 2 * Math.PI, 1.5);
            } else {
                dataset[i] = generateRandomWalk(seriesLength, random);
            }
        }
        
        double[] query = generateSineWave(seriesLength, 1.0, 0.0, 1.0);
        
        // Busca k-NN otimizada
        DTW optimizedDTW = new DTW(7); // Sakoe-Chiba band
        DTWNeighbors neighbors = new DTWNeighbors(optimizedDTW, true, 4, true);
        
        int k = 5;
        long startTime = System.nanoTime();
        List<DTWNeighbors.NeighborResult> results = neighbors.kNearest(query, dataset, k);
        long endTime = System.nanoTime();
        
        logger.info("🔍 Busca k-NN: {} vizinhos em dataset de {} séries", k, datasetSize);
        logger.info("⏱️  Tempo: {:.3f}ms", (endTime - startTime) / 1_000_000.0);
        
        logger.info("🏆 Top {} vizinhos mais próximos:", k);
        for (int i = 0; i < results.size(); i++) {
            DTWNeighbors.NeighborResult result = results.get(i);
            String type = result.getIndex() < 30 ? "Seno" : 
                         result.getIndex() < 60 ? "Cosseno" : "RandomWalk";
            logger.info("   {}. Índice: {} ({}), Distância: {:.4f}", 
                       i + 1, result.getIndex(), type, result.getDistance());
        }
        
        // Estatísticas de otimização
        logger.info("📊 Lower bound pruning: {:.1f}%% das comparações evitadas", 
                   neighbors.getStats().getPruningRate() * 100);
    }
    
    private static void demonstrateCompletePipeline() {
        logger.info("\n🟡 4. Pipeline Completo de Análise");
        logger.info("─".repeat(60));
        
        // Simular dados reais: múltiplas séries com ruído
        Random random = new Random(789);
        int numSeries = 30;
        int seriesLength = 60;
        
        double[][] rawData = new double[numSeries][];
        
        // Gerar 3 grupos com padrões diferentes + ruído
        for (int i = 0; i < numSeries; i++) {
            double[] baseSeries;
            if (i < 10) {
                // Grupo 1: Tendência crescente
                baseSeries = generateTrendSeries(seriesLength, 0.05, random);
            } else if (i < 20) {
                // Grupo 2: Padrão sazonal
                baseSeries = generateSeasonalSeries(seriesLength, 4, random);
            } else {
                // Grupo 3: Padrão estacionário
                baseSeries = generateStationarySeries(seriesLength, random);
            }
            rawData[i] = baseSeries;
        }
        
        logger.info("📂 Dataset sintético: {} séries de {} pontos", numSeries, seriesLength);
        
        // 1. Preprocessamento
        long startTime = System.currentTimeMillis();
        TimeSeriesScalerMeanVariance scaler = new TimeSeriesScalerMeanVariance();
        RealMatrix[] scaledData = scaler.fitTransform(MatrixUtils.toTimeSeriesDataset(rawData));
        long prepTime = System.currentTimeMillis() - startTime;
        
        // 2. Clustering
        startTime = System.currentTimeMillis();
        KShape kshape = new KShape(3, 50, 1e-4, 3, false, 42L, "random");
        int[] labels = kshape.fitPredict(scaledData);
        long clusterTime = System.currentTimeMillis() - startTime;
        
        // 3. Avaliação com DTW
        startTime = System.currentTimeMillis();
        DTW dtw = new DTW(10);
        double totalIntraClusterDistance = 0.0;
        int[] clusterSizes = new int[3];
        
        for (int c = 0; c < 3; c++) {
            double clusterSum = 0.0;
            int clusterCount = 0;
            
            for (int i = 0; i < labels.length; i++) {
                if (labels[i] == c) {
                    clusterSizes[c]++;
                    for (int j = i + 1; j < labels.length; j++) {
                        if (labels[j] == c) {
                            clusterSum += dtw.distance(rawData[i], rawData[j]);
                            clusterCount++;
                        }
                    }
                }
            }
            
            if (clusterCount > 0) {
                totalIntraClusterDistance += clusterSum / clusterCount;
            }
        }
        long evalTime = System.currentTimeMillis() - startTime;
        
        // Resultados
        logger.info("⚙️  Pipeline completo executado:");
        logger.info("   1. Preprocessamento: {}ms", prepTime);
        logger.info("   2. Clustering:       {}ms (convergiu em {} iterações)", 
                   clusterTime, kshape.getNIter());
        logger.info("   3. Avaliação DTW:    {}ms", evalTime);
        logger.info("   Total:               {}ms", prepTime + clusterTime + evalTime);
        
        logger.info("📊 Resultados finais:");
        logger.info("   Clusters encontrados: C0={} C1={} C2={}", 
                   clusterSizes[0], clusterSizes[1], clusterSizes[2]);
        logger.info("   Inércia KShape: {:.6f}", kshape.getInertia());
        logger.info("   Distância intra-cluster média (DTW): {:.6f}", totalIntraClusterDistance / 3);
        
        // Verificar qualidade do clustering
        boolean goodClustering = clusterSizes[0] > 0 && clusterSizes[1] > 0 && clusterSizes[2] > 0;
        logger.info("   Qualidade: {} (todos os clusters têm elementos)", 
                   goodClustering ? "✅ BOM" : "❌ RUIM");
    }
    
    // Métodos auxiliares para geração de dados
    
    private static double[] generateSineWave(int length, double amplitude, double phase, double frequency) {
        double[] series = new double[length];
        for (int i = 0; i < length; i++) {
            series[i] = amplitude * Math.sin(frequency * 2 * Math.PI * i / length + phase);
        }
        return series;
    }
    
    private static double[] generateCosineWave(int length, double amplitude, double phase, double frequency) {
        double[] series = new double[length];
        for (int i = 0; i < length; i++) {
            series[i] = amplitude * Math.cos(frequency * 2 * Math.PI * i / length + phase);
        }
        return series;
    }
    
    private static double[] generateRandomWalk(int length, Random random) {
        double[] series = new double[length];
        series[0] = random.nextGaussian();
        for (int i = 1; i < length; i++) {
            series[i] = series[i-1] + random.nextGaussian() * 0.1;
        }
        return series;
    }
    
    private static double[] generateTrendSeries(int length, double slope, Random random) {
        double[] series = new double[length];
        for (int i = 0; i < length; i++) {
            series[i] = slope * i + random.nextGaussian() * 0.1;
        }
        return series;
    }
    
    private static double[] generateSeasonalSeries(int length, double frequency, Random random) {
        double[] series = new double[length];
        for (int i = 0; i < length; i++) {
            series[i] = Math.sin(frequency * 2 * Math.PI * i / length) + random.nextGaussian() * 0.1;
        }
        return series;
    }
    
    private static double[] generateStationarySeries(int length, Random random) {
        double[] series = new double[length];
        for (int i = 0; i < length; i++) {
            series[i] = random.nextGaussian() * 0.5;
        }
        return series;
    }
}
