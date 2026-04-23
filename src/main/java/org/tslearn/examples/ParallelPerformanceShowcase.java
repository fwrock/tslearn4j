package org.tslearn.examples;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tslearn.clustering.ParallelTimeSeriesKMeans;
import org.tslearn.clustering.TimeSeriesKMeans;
import org.tslearn.matrix_profile.MatrixProfile;
import org.tslearn.matrix_profile.ParallelMatrixProfile;
import org.tslearn.metrics.DTW;
import org.tslearn.metrics.ParallelDTW;
import org.tslearn.utils.ParallelUtils;

import java.util.Arrays;
import java.util.Random;

/**
 * Demonstração completa das melhorias de performance obtidas com paralelização
 * nos algoritmos de machine learning para séries temporais.
 * 
 * Este exemplo compara:
 * - K-means sequencial vs paralelo
 * - DTW sequencial vs paralelo  
 * - Matrix Profile sequencial vs paralelo
 * - Diferentes configurações de paralelização
 * - Análise de escalabilidade
 * 
 * @author TSLearn4J
 */
public class ParallelPerformanceShowcase {
    
    private static final Logger logger = LoggerFactory.getLogger(ParallelPerformanceShowcase.class);
    
    public static void main(String[] args) {
        logger.info("=== TSLearn4J Parallel Performance Showcase ===");
        
        ParallelPerformanceShowcase showcase = new ParallelPerformanceShowcase();
        
        // Executar benchmarks com diferentes tamanhos de dados
        int[] dataSizes = {100, 500, 1000, 2000, 5000};
        
        for (int dataSize : dataSizes) {
            logger.info("\n--- Benchmark com {} amostras ---", dataSize);
            showcase.runBenchmark(dataSize);
        }
        
        // Análise detalhada de escalabilidade
        logger.info("\n=== Análise de Escalabilidade ===");
        showcase.runScalabilityAnalysis();
        
        // Comparação de configurações de paralelização
        logger.info("\n=== Comparação de Configurações ===");
        showcase.compareParallelConfigurations();
        
        logger.info("\n=== Showcase Concluído ===");
    }
    
    /**
     * Executa benchmark completo para um tamanho específico de dados.
     */
    public void runBenchmark(int dataSize) {
        // Gerar dados sintéticos
        double[][][] dataset = generateSyntheticDataset(dataSize, 100, 1);
        double[] timeSeries = generateTimeSeries(dataSize * 10, 0.1);
        
        logger.info("Dataset gerado: {} amostras, {} pontos por série", dataSize, 100);
        
        // Benchmark K-means
        benchmarkKMeans(dataset);
        
        // Benchmark DTW
        benchmarkDTW(dataset);
        
        // Benchmark Matrix Profile
        benchmarkMatrixProfile(timeSeries);
    }
    
    /**
     * Benchmark do K-means paralelo vs sequencial.
     */
    private void benchmarkKMeans(double[][][] dataset) {
        logger.info("\n-- K-means Benchmark --");
        
        int nClusters = 5;
        int nInit = 5;
        
        // K-means sequencial
        long startTime = System.currentTimeMillis();
        
        TimeSeriesKMeans sequentialKMeans = new TimeSeriesKMeans.Builder()
                .nClusters(nClusters)
                .nInit(nInit)
                .maxIter(50)
                .metric(TimeSeriesKMeans.Metric.EUCLIDEAN)
                .verbose(false)
                .build();
        
        sequentialKMeans.fit(dataset);
        long sequentialTime = System.currentTimeMillis() - startTime;
        
        // K-means paralelo
        startTime = System.currentTimeMillis();
        
        ParallelTimeSeriesKMeans parallelKMeans = new ParallelTimeSeriesKMeans.Builder()
                .nClusters(nClusters)
                .nInit(nInit)
                .maxIter(50)
                .metric(ParallelTimeSeriesKMeans.Metric.EUCLIDEAN)
                .autoConfigureParallelism(dataset.length)
                .verbose(false)
                .build();
        
        parallelKMeans.fit(dataset);
        long parallelTime = System.currentTimeMillis() - startTime;
        
        // Análise dos resultados
        double sequentialInertia = sequentialKMeans.getInertia();
        double parallelInertia = parallelKMeans.getInertia();
        double speedup = (double) sequentialTime / parallelTime;
        
        logger.info("K-means Sequencial: {}ms, inércia: {:.6f}", sequentialTime, sequentialInertia);
        logger.info("K-means Paralelo:   {}ms, inércia: {:.6f}", parallelTime, parallelInertia);
        logger.info("Speedup: {:.2f}x", speedup);
        logger.info("Diferença de inércia: {:.6f}", Math.abs(sequentialInertia - parallelInertia));
        
        parallelKMeans.close();
    }
    
    /**
     * Benchmark do DTW paralelo vs sequencial.
     */
    private void benchmarkDTW(double[][][] dataset) {
        logger.info("\n-- DTW Benchmark --");
        
        // Extrair duas séries para comparação
        double[][] ts1 = dataset[0];
        double[][] ts2 = dataset[Math.min(1, dataset.length - 1)];
        double[] series1 = flattenTimeSeries(ts1);
        double[] series2 = flattenTimeSeries(ts2);
        
        // DTW sequencial
        DTW sequentialDTW = new DTW();
        long startTime = System.currentTimeMillis();
        double sequentialDistance = sequentialDTW.distance(series1, series2);
        long sequentialTime = System.currentTimeMillis() - startTime;
        
        // DTW paralelo
        ParallelDTW parallelDTW = new ParallelDTW();
        startTime = System.currentTimeMillis();
        double parallelDistance = parallelDTW.distance(series1, series2);
        long parallelTime = System.currentTimeMillis() - startTime;
        
        double speedup = (double) sequentialTime / parallelTime;
        
        logger.info("DTW Sequencial: {}ms, distância: {:.6f}", sequentialTime, sequentialDistance);
        logger.info("DTW Paralelo:   {}ms, distância: {:.6f}", parallelTime, parallelDistance);
        logger.info("Speedup: {:.2f}x", speedup);
        logger.info("Diferença de distância: {:.6f}", Math.abs(sequentialDistance - parallelDistance));
        
        // Benchmark de matriz de distâncias
        if (dataset.length >= 10) {
            benchmarkDTWMatrix(dataset);
        }
    }
    
    /**
     * Benchmark de matriz de distâncias DTW.
     */
    private void benchmarkDTWMatrix(double[][][] dataset) {
        logger.info("\n-- DTW Distance Matrix Benchmark --");
        
        // Usar subconjunto para matriz de distâncias
        int matrixSize = Math.min(50, dataset.length);
        double[][] flatSeries = new double[matrixSize][];
        
        for (int i = 0; i < matrixSize; i++) {
            flatSeries[i] = flattenTimeSeries(dataset[i]);
        }
        
        // Sequencial
        DTW sequentialDTW = new DTW();
        long startTime = System.currentTimeMillis();
        
        double[][] sequentialMatrix = new double[matrixSize][matrixSize];
        for (int i = 0; i < matrixSize; i++) {
            for (int j = i; j < matrixSize; j++) {
                double dist = sequentialDTW.distance(flatSeries[i], flatSeries[j]);
                sequentialMatrix[i][j] = dist;
                sequentialMatrix[j][i] = dist;
            }
        }
        
        long sequentialTime = System.currentTimeMillis() - startTime;
        
        // Paralelo
        ParallelDTW parallelDTW = new ParallelDTW();
        startTime = System.currentTimeMillis();
        double[][] parallelMatrix = parallelDTW.distanceMatrixSquare(flatSeries);
        long parallelTime = System.currentTimeMillis() - startTime;
        
        double speedup = (double) sequentialTime / parallelTime;
        
        logger.info("Matriz DTW {}x{} Sequencial: {}ms", matrixSize, matrixSize, sequentialTime);
        logger.info("Matriz DTW {}x{} Paralelo:   {}ms", matrixSize, matrixSize, parallelTime);
        logger.info("Speedup: {:.2f}x", speedup);
        
        // Verificar correção
        double maxDiff = 0.0;
        for (int i = 0; i < matrixSize; i++) {
            for (int j = 0; j < matrixSize; j++) {
                maxDiff = Math.max(maxDiff, Math.abs(sequentialMatrix[i][j] - parallelMatrix[i][j]));
            }
        }
        logger.info("Diferença máxima: {:.6f}", maxDiff);
    }
    
    /**
     * Benchmark do Matrix Profile paralelo vs sequencial.
     */
    private void benchmarkMatrixProfile(double[] timeSeries) {
        logger.info("\n-- Matrix Profile Benchmark --");
        
        int subsequenceLength = 20;
        
        // Matrix Profile sequencial
        MatrixProfile sequentialMP = new MatrixProfile.Builder()
                .subsequenceLength(subsequenceLength)
                .normalize(true)
                .verbose(false)
                .build();
        
        long startTime = System.currentTimeMillis();
        MatrixProfile.MatrixProfileResult sequentialResult = sequentialMP.stamp(timeSeries);
        long sequentialTime = System.currentTimeMillis() - startTime;
        
        // Matrix Profile paralelo
        ParallelMatrixProfile parallelMP = new ParallelMatrixProfile.ParallelBuilder()
                .subsequenceLength(subsequenceLength)
                .normalize(true)
                .autoConfigureParallelism(timeSeries.length)
                .verbose(false)
                .build();
        
        startTime = System.currentTimeMillis();
        MatrixProfile.MatrixProfileResult parallelResult = parallelMP.stamp(timeSeries);
        long parallelTime = System.currentTimeMillis() - startTime;
        
        double speedup = (double) sequentialTime / parallelTime;
        
        logger.info("Matrix Profile Sequencial: {}ms", sequentialTime);
        logger.info("Matrix Profile Paralelo:   {}ms", parallelTime);
        logger.info("Speedup: {:.2f}x", speedup);
        
        // Verificar correção comparando alguns valores
        double[] seqMP = sequentialResult.getMatrixProfile();
        double[] parMP = parallelResult.getMatrixProfile();
        
        if (seqMP.length == parMP.length) {
            double maxDiff = 0.0;
            for (int i = 0; i < Math.min(100, seqMP.length); i++) {
                maxDiff = Math.max(maxDiff, Math.abs(seqMP[i] - parMP[i]));
            }
            logger.info("Diferença máxima (amostra): {:.6f}", maxDiff);
        }
    }
    
    /**
     * Análise de escalabilidade com diferentes números de threads.
     */
    private void runScalabilityAnalysis() {
        int dataSize = 1000;
        double[][][] dataset = generateSyntheticDataset(dataSize, 100, 1);
        
        int[] threadCounts = {1, 2, 4, 8, 16};
        
        logger.info("Análise de escalabilidade K-means (dataSize={})", dataSize);
        logger.info("Threads\tTempo(ms)\tSpeedup\tEficiência");
        
        long baselineTime = 0;
        
        for (int threads : threadCounts) {
            ParallelUtils.ParallelConfig config = new ParallelUtils.ParallelConfig(threads, 100, 50);
            
            ParallelTimeSeriesKMeans kmeans = new ParallelTimeSeriesKMeans.Builder()
                    .nClusters(5)
                    .nInit(3)
                    .maxIter(30)
                    .metric(ParallelTimeSeriesKMeans.Metric.EUCLIDEAN)
                    .parallelConfig(config)
                    .verbose(false)
                    .build();
            
            long startTime = System.currentTimeMillis();
            kmeans.fit(dataset);
            long executionTime = System.currentTimeMillis() - startTime;
            
            if (threads == 1) {
                baselineTime = executionTime;
            }
            
            double speedup = (double) baselineTime / executionTime;
            double efficiency = speedup / threads;
            
            logger.info("{}\t{}\t{:.2f}\t{:.2f}", threads, executionTime, speedup, efficiency);
            
            kmeans.close();
        }
    }
    
    /**
     * Comparação de diferentes configurações de paralelização.
     */
    private void compareParallelConfigurations() {
        int dataSize = 2000;
        double[][][] dataset = generateSyntheticDataset(dataSize, 100, 1);
        
        logger.info("Comparação de configurações para {} amostras", dataSize);
        
        // Configuração padrão
        ParallelUtils.ParallelConfig defaultConfig = ParallelUtils.ParallelConfig.defaultConfig();
        long defaultTime = benchmarkConfiguration(dataset, defaultConfig, "Padrão");
        
        // Configuração otimizada para tamanho dos dados
        ParallelUtils.ParallelConfig optimizedConfig = ParallelUtils.ParallelConfig.forDataSize(dataSize);
        long optimizedTime = benchmarkConfiguration(dataset, optimizedConfig, "Otimizada");
        
        // Configuração conservadora
        ParallelUtils.ParallelConfig conservativeConfig = new ParallelUtils.ParallelConfig(2, 200, 100);
        long conservativeTime = benchmarkConfiguration(dataset, conservativeConfig, "Conservadora");
        
        // Configuração agressiva
        ParallelUtils.ParallelConfig aggressiveConfig = new ParallelUtils.ParallelConfig(
                Runtime.getRuntime().availableProcessors() * 2, 50, 25);
        long aggressiveTime = benchmarkConfiguration(dataset, aggressiveConfig, "Agressiva");
        
        logger.info("\nMelhor configuração: {}", 
                   findBestConfiguration(defaultTime, optimizedTime, conservativeTime, aggressiveTime));
    }
    
    /**
     * Benchmark de uma configuração específica.
     */
    private long benchmarkConfiguration(double[][][] dataset, ParallelUtils.ParallelConfig config, String name) {
        ParallelTimeSeriesKMeans kmeans = new ParallelTimeSeriesKMeans.Builder()
                .nClusters(5)
                .nInit(3)
                .maxIter(30)
                .metric(ParallelTimeSeriesKMeans.Metric.EUCLIDEAN)
                .parallelConfig(config)
                .verbose(false)
                .build();
        
        long startTime = System.currentTimeMillis();
        kmeans.fit(dataset);
        long executionTime = System.currentTimeMillis() - startTime;
        
        logger.info("{}: {}ms (parallelism={}, threshold={}, chunk={})", 
                   name, executionTime, config.getParallelism(), 
                   config.getMinThreshold(), config.getChunkSize());
        
        kmeans.close();
        return executionTime;
    }
    
    /**
     * Encontra a melhor configuração baseada nos tempos.
     */
    private String findBestConfiguration(long defaultTime, long optimizedTime, 
                                       long conservativeTime, long aggressiveTime) {
        
        long bestTime = Math.min(Math.min(defaultTime, optimizedTime), 
                                Math.min(conservativeTime, aggressiveTime));
        
        if (bestTime == defaultTime) return "Padrão";
        if (bestTime == optimizedTime) return "Otimizada";
        if (bestTime == conservativeTime) return "Conservadora";
        return "Agressiva";
    }
    
    /**
     * Gera dataset sintético para testes.
     */
    private double[][][] generateSyntheticDataset(int nSamples, int timeLength, int nFeatures) {
        Random random = new Random(42);
        double[][][] dataset = new double[nSamples][timeLength][nFeatures];
        
        for (int i = 0; i < nSamples; i++) {
            // Gerar série com padrão baseado no cluster
            int cluster = i % 5;
            double amplitude = 1.0 + cluster * 0.5;
            double frequency = 0.1 + cluster * 0.02;
            double phase = cluster * Math.PI / 4;
            
            for (int t = 0; t < timeLength; t++) {
                for (int d = 0; d < nFeatures; d++) {
                    double signal = amplitude * Math.sin(2 * Math.PI * frequency * t + phase);
                    double noise = random.nextGaussian() * 0.1;
                    dataset[i][t][d] = signal + noise;
                }
            }
        }
        
        return dataset;
    }
    
    /**
     * Gera série temporal sintética.
     */
    private double[] generateTimeSeries(int length, double noiseLevel) {
        Random random = new Random(42);
        double[] series = new double[length];
        
        for (int i = 0; i < length; i++) {
            // Componente de tendência
            double trend = 0.001 * i;
            
            // Componentes sazonais
            double seasonal1 = Math.sin(2 * Math.PI * i / 100.0);
            double seasonal2 = 0.5 * Math.cos(2 * Math.PI * i / 50.0);
            
            // Ruído
            double noise = random.nextGaussian() * noiseLevel;
            
            series[i] = trend + seasonal1 + seasonal2 + noise;
        }
        
        return series;
    }
    
    /**
     * Converte série temporal multidimensional para unidimensional.
     */
    private double[] flattenTimeSeries(double[][] timeSeries) {
        double[] flattened = new double[timeSeries.length];
        for (int i = 0; i < timeSeries.length; i++) {
            flattened[i] = timeSeries[i][0]; // Usar apenas a primeira dimensão
        }
        return flattened;
    }
}
