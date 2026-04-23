package org.tslearn.clustering;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;

import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.random.Well19937c;
import org.apache.commons.math3.util.FastMath;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tslearn.barycenters.DTWBarycenter;
import org.tslearn.barycenters.EuclideanBarycenter;
import org.tslearn.barycenters.ParallelDTWBarycenter;
import org.tslearn.metrics.DTW;
import org.tslearn.utils.ArrayUtils;
import org.tslearn.utils.ParallelUtils;

/**
 * Versão paralelizada do K-means clustering para dados de séries temporais.
 * 
 * Esta implementação otimizada oferece:
 * - Paralelização automática baseada no tamanho dos dados
 * - Inicializações múltiplas executadas em paralelo
 * - Cálculo paralelo de distâncias e atribuições de cluster
 * - Computação paralela de baricentros DTW
 * - Balanceamento adaptivo de carga
 * - Otimizações específicas para diferentes métricas
 * 
 * @author TSLearn4J
 */
public class ParallelTimeSeriesKMeans {
    
    private static final Logger logger = LoggerFactory.getLogger(ParallelTimeSeriesKMeans.class);
    
    public enum Metric {
        EUCLIDEAN,
        DTW,
        SOFT_DTW
    }
    
    public enum InitMethod {
        KMEANS_PLUS_PLUS,
        RANDOM,
        CUSTOM
    }
    
    // Parâmetros do algoritmo
    private final int nClusters;
    private final int maxIter;
    private final double tolerance;
    private final int nInit;
    private final Metric metric;
    private final int maxIterBarycenter;
    private final InitMethod initMethod;
    private final boolean verbose;
    private final boolean dtwInertia;
    private final RandomGenerator random;
    private final ParallelUtils.ParallelConfig parallelConfig;
    
    // Parâmetros específicos das métricas
    private final Map<String, Object> metricParams;
    
    // Resultados do treinamento
    private double[][][] clusterCenters;
    private int[] labels;
    private double inertia;
    private int nIter;
    private boolean fitted = false;
    
    // Componentes para paralelização
    private final DTW dtw;
    private final ParallelUtils.TimeSeriesWorker parallelWorker;
    private final ForkJoinPool customPool;
    
    /**
     * Builder pattern para configuração flexível.
     */
    public static class Builder {
        private int nClusters = 3;
        private int maxIter = 100;
        private double tolerance = 1e-6;
        private int nInit = 10;
        private Metric metric = Metric.EUCLIDEAN;
        private int maxIterBarycenter = 100;
        private InitMethod initMethod = InitMethod.KMEANS_PLUS_PLUS;
        private boolean verbose = false;
        private boolean dtwInertia = false;
        private Long randomSeed = null;
        private Map<String, Object> metricParams = new HashMap<>();
        private ParallelUtils.ParallelConfig parallelConfig = ParallelUtils.ParallelConfig.defaultConfig();
        
        public Builder nClusters(int nClusters) {
            this.nClusters = nClusters;
            return this;
        }
        
        public Builder maxIter(int maxIter) {
            this.maxIter = maxIter;
            return this;
        }
        
        public Builder tolerance(double tolerance) {
            this.tolerance = tolerance;
            return this;
        }
        
        public Builder nInit(int nInit) {
            this.nInit = nInit;
            return this;
        }
        
        public Builder metric(Metric metric) {
            this.metric = metric;
            return this;
        }
        
        public Builder maxIterBarycenter(int maxIterBarycenter) {
            this.maxIterBarycenter = maxIterBarycenter;
            return this;
        }
        
        public Builder initMethod(InitMethod initMethod) {
            this.initMethod = initMethod;
            return this;
        }
        
        public Builder verbose(boolean verbose) {
            this.verbose = verbose;
            return this;
        }
        
        public Builder dtwInertia(boolean dtwInertia) {
            this.dtwInertia = dtwInertia;
            return this;
        }
        
        public Builder randomSeed(long randomSeed) {
            this.randomSeed = randomSeed;
            return this;
        }
        
        public Builder metricParams(Map<String, Object> metricParams) {
            this.metricParams = new HashMap<>(metricParams);
            return this;
        }
        
        public Builder parallelConfig(ParallelUtils.ParallelConfig parallelConfig) {
            this.parallelConfig = parallelConfig;
            return this;
        }
        
        public Builder autoConfigureParallelism(int dataSize) {
            this.parallelConfig = ParallelUtils.ParallelConfig.forDataSize(dataSize);
            return this;
        }
        
        public ParallelTimeSeriesKMeans build() {
            return new ParallelTimeSeriesKMeans(this);
        }
    }
    
    private ParallelTimeSeriesKMeans(Builder builder) {
        this.nClusters = builder.nClusters;
        this.maxIter = builder.maxIter;
        this.tolerance = builder.tolerance;
        this.nInit = builder.nInit;
        this.metric = builder.metric;
        this.maxIterBarycenter = builder.maxIterBarycenter;
        this.initMethod = builder.initMethod;
        this.verbose = builder.verbose;
        this.dtwInertia = builder.dtwInertia;
        this.metricParams = builder.metricParams;
        this.parallelConfig = builder.parallelConfig;
        
        // Inicializar componentes
        this.random = builder.randomSeed != null ? 
                     new Well19937c(builder.randomSeed) : new Well19937c();
        
        this.dtw = createDTWInstance();
        this.parallelWorker = new ParallelUtils.TimeSeriesWorker(parallelConfig);
        this.customPool = new ForkJoinPool(parallelConfig.getParallelism());
        
        // Configurar shutdown hook
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            if (!customPool.isShutdown()) {
                customPool.shutdown();
            }
        }));
    }
    
    private DTW createDTWInstance() {
        if (metricParams.containsKey("global_constraint")) {
            String constraint = (String) metricParams.get("global_constraint");
            if ("sakoe_chiba".equals(constraint)) {
                int sakoeChiba = (Integer) metricParams.getOrDefault("sakoe_chiba_radius", 1);
                return new DTW(sakoeChiba);
            }
        }
        return new DTW();
    }
    
    /**
     * Treina o modelo K-means paralelizando múltiplas inicializações.
     */
    public ParallelTimeSeriesKMeans fit(double[][][] X) {
        if (X == null || X.length == 0) {
            throw new IllegalArgumentException("Dataset não pode ser vazio");
        }
        
        long startTime = System.currentTimeMillis();
        if (verbose) {
            logger.info("Iniciando treinamento paralelo: {} amostras, {} clusters, {} inicializações", 
                       X.length, nClusters, nInit);
        }
        
        try {
            if (nInit == 1) {
                // Uma única inicialização - executar sequencialmente
                fitSingleInit(X);
            } else if (parallelConfig.shouldParallelize(nInit)) {
                // Múltiplas inicializações - executar em paralelo
                fitMultipleInitsParallel(X);
            } else {
                // Múltiplas inicializações - executar sequencialmente
                fitMultipleInitsSequential(X);
            }
            
            this.fitted = true;
            
            if (verbose) {
                long duration = System.currentTimeMillis() - startTime;
                logger.info("Treinamento concluído em {}ms: inércia final = {:.6f}, {} iterações", 
                           duration, inertia, nIter);
            }
            
        } catch (Exception e) {
            logger.error("Erro durante o treinamento", e);
            throw new RuntimeException("Falha no treinamento", e);
        }
        
        return this;
    }
    
    /**
     * Executa múltiplas inicializações em paralelo.
     */
    private void fitMultipleInitsParallel(double[][][] X) {
        if (verbose) {
            logger.info("Executando {} inicializações em paralelo com {} threads", 
                       nInit, parallelConfig.getParallelism());
        }
        
        // Criar tasks para cada inicialização
        List<CompletableFuture<InitResult>> futures = new ArrayList<>();
        
        for (int init = 0; init < nInit; init++) {
            final int initIndex = init;
            final long seed = random.nextLong(); // Seed única para cada inicialização
            
            CompletableFuture<InitResult> future = CompletableFuture.supplyAsync(() -> {
                try {
                    return executeOneInit(X, initIndex, seed);
                } catch (Exception e) {
                    if (verbose) {
                        logger.warn("Inicialização {} falhou: {}", initIndex + 1, e.getMessage());
                    }
                    return null;
                }
            }, customPool);
            
            futures.add(future);
        }
        
        // Coletar resultados e encontrar o melhor
        InitResult bestResult = null;
        double bestInertia = Double.POSITIVE_INFINITY;
        
        for (int i = 0; i < futures.size(); i++) {
            try {
                InitResult result = futures.get(i).get();
                if (result != null && result.inertia < bestInertia) {
                    bestInertia = result.inertia;
                    bestResult = result;
                    
                    if (verbose) {
                        logger.info("Nova melhor inércia da inicialização {}: {:.6f}", 
                                   i + 1, bestInertia);
                    }
                }
            } catch (Exception e) {
                if (verbose) {
                    logger.warn("Erro ao obter resultado da inicialização {}: {}", i + 1, e.getMessage());
                }
            }
        }
        
        if (bestResult == null) {
            throw new RuntimeException("Todas as inicializações falharam");
        }
        
        // Definir os melhores resultados
        this.clusterCenters = bestResult.clusterCenters;
        this.inertia = bestResult.inertia;
        this.nIter = bestResult.nIter;
        this.labels = bestResult.labels;
    }
    
    /**
     * Executa múltiplas inicializações sequencialmente.
     */
    private void fitMultipleInitsSequential(double[][][] X) {
        double bestInertia = Double.POSITIVE_INFINITY;
        double[][][] bestCenters = null;
        int bestNIter = 0;
        int[] bestLabels = null;
        
        for (int init = 0; init < nInit; init++) {
            if (verbose && nInit > 1) {
                logger.info("Inicialização {} de {}", init + 1, nInit);
            }
            
            try {
                long seed = random.nextLong();
                InitResult result = executeOneInit(X, init, seed);
                
                if (result.inertia < bestInertia) {
                    bestInertia = result.inertia;
                    bestCenters = result.clusterCenters;
                    bestNIter = result.nIter;
                    bestLabels = result.labels;
                    
                    if (verbose) {
                        logger.info("Nova melhor inércia: {:.6f}", bestInertia);
                    }
                }
            } catch (Exception e) {
                if (verbose) {
                    logger.warn("Inicialização {} falhou: {}", init + 1, e.getMessage());
                }
            }
        }
        
        this.clusterCenters = bestCenters;
        this.inertia = bestInertia;
        this.nIter = bestNIter;
        this.labels = bestLabels;
    }
    
    /**
     * Executa uma única inicialização.
     */
    private void fitSingleInit(double[][][] X) {
        InitResult result = executeOneInit(X, 0, random.nextLong());
        this.clusterCenters = result.clusterCenters;
        this.inertia = result.inertia;
        this.nIter = result.nIter;
        this.labels = result.labels;
    }
    
    /**
     * Resultado de uma inicialização.
     */
    private static class InitResult {
        final double[][][] clusterCenters;
        final double inertia;
        final int nIter;
        final int[] labels;
        
        InitResult(double[][][] clusterCenters, double inertia, int nIter, int[] labels) {
            this.clusterCenters = clusterCenters;
            this.inertia = inertia;
            this.nIter = nIter;
            this.labels = labels;
        }
    }
    
    /**
     * Executa uma única inicialização do algoritmo K-means.
     */
    private InitResult executeOneInit(double[][][] X, int initIndex, long seed) {
        RandomGenerator localRandom = new Well19937c(seed);
        int nSamples = X.length;
        int timeLength = X[0].length;
        int nFeatures = X[0][0].length;
        
        // Inicializar centróides
        double[][][] localCenters = initializeCentroids(X, localRandom);
        int[] localLabels = new int[nSamples];
        
        double oldInertia = Double.POSITIVE_INFINITY;
        int iter;
        
        for (iter = 0; iter < maxIter; iter++) {
            // Atribuir amostras aos clusters (paralelo)
            double localInertia = assignSamplesToClustersParallel(X, localCenters, localLabels);
            
            if (verbose && initIndex == 0) {
                System.out.printf("Iter %d: %.6f --> ", iter, localInertia);
            }
            
            // Atualizar centróides (paralelo se necessário)
            updateCentroidsParallel(X, localCenters, localLabels);
            
            // Verificar convergência
            if (FastMath.abs(oldInertia - localInertia) < tolerance) {
                if (verbose && initIndex == 0) {
                    System.out.println("Convergiu");
                }
                break;
            }
            oldInertia = localInertia;
        }
        
        if (verbose && initIndex == 0) {
            System.out.println();
        }
        
        // Calcular inércia final
        double finalInertia = calculateInertia(X, localCenters, localLabels);
        
        return new InitResult(localCenters, finalInertia, iter + 1, localLabels);
    }
    
    /**
     * Atribui amostras aos clusters usando paralelização adaptiva.
     */
    private double assignSamplesToClustersParallel(double[][][] X, double[][][] centers, int[] labels) {
        ParallelUtils.DistanceFunction distanceFunc = this::computeDistance;
        
        // Usar worker paralelo para atribuição
        int[] newLabels = parallelWorker.assignSamplesToClusters(X, centers, distanceFunc);
        System.arraycopy(newLabels, 0, labels, 0, newLabels.length);
        
        // Calcular inércia total em paralelo
        double[] distances = ParallelUtils.parallelMapToDouble(X.length, i -> 
            computeDistance(X[i], centers[labels[i]])
        );
        
        return Arrays.stream(distances).sum();
    }
    
    /**
     * Atualiza centróides usando paralelização quando benéfico.
     */
    private void updateCentroidsParallel(double[][][] X, double[][][] centers, int[] labels) {
        int nSamples = X.length;
        
        // Determinar se deve paralelizar baseado no número de clusters
        boolean shouldParallelizeClusters = parallelConfig.shouldParallelize(nClusters);
        
        if (shouldParallelizeClusters) {
            // Paralelizar por cluster
            IntStream.range(0, nClusters)
                    .parallel()
                    .forEach(k -> updateSingleCentroid(X, centers, labels, k));
        } else {
            // Executar sequencialmente
            for (int k = 0; k < nClusters; k++) {
                updateSingleCentroid(X, centers, labels, k);
            }
        }
    }
    
    /**
     * Atualiza um único centróide.
     */
    private void updateSingleCentroid(double[][][] X, double[][][] centers, int[] labels, int k) {
        int nSamples = X.length;
        
        // Coletar amostras do cluster k
        List<double[][]> clusterSamples = new ArrayList<>();
        for (int i = 0; i < nSamples; i++) {
            if (labels[i] == k) {
                clusterSamples.add(X[i]);
            }
        }
        
        if (clusterSamples.isEmpty()) {
            return; // Cluster vazio, manter centróide atual
        }
        
        // Calcular novo centróide baseado na métrica
        double[][][] samplesArray = clusterSamples.toArray(double[][][]::new);
        
        switch (metric) {
            case EUCLIDEAN -> centers[k] = EuclideanBarycenter.compute(samplesArray);
            case DTW -> {
                // Usar versão paralelizada do DTW barycenter se disponível
                if (samplesArray.length > 10 && parallelConfig.shouldParallelize(samplesArray.length)) {
                    centers[k] = ParallelDTWBarycenter.compute(samplesArray, centers[k], maxIterBarycenter);
                } else {
                    centers[k] = DTWBarycenter.compute(samplesArray, centers[k], maxIterBarycenter);
                }
            }
            case SOFT_DTW -> throw new UnsupportedOperationException("SoftDTW não implementado ainda");
            default -> throw new IllegalArgumentException("Métrica não suportada: " + metric);
        }
    }
    
    /**
     * Calcula a distância entre duas séries temporais.
     */
    private double computeDistance(double[][] ts1, double[][] ts2) {
        return switch (metric) {
            case EUCLIDEAN -> computeEuclideanDistance(ts1, ts2);
            case DTW -> dtw.distance(ts1, ts2);
            case SOFT_DTW -> throw new UnsupportedOperationException("SoftDTW não implementado ainda");
        };
    }
    
    /**
     * Calcula a distância euclidiana entre duas séries temporais.
     */
    private double computeEuclideanDistance(double[][] ts1, double[][] ts2) {
        if (ts1.length != ts2.length) {
            throw new IllegalArgumentException("Séries temporais devem ter o mesmo comprimento");
        }
        
        double sum = 0.0;
        for (int t = 0; t < ts1.length; t++) {
            for (int d = 0; d < ts1[t].length; d++) {
                double diff = ts1[t][d] - ts2[t][d];
                sum += diff * diff;
            }
        }
        return Math.sqrt(sum);
    }
    
    /**
     * Calcula a inércia total do clustering.
     */
    private double calculateInertia(double[][][] X, double[][][] centers, int[] labels) {
        double[] distances = ParallelUtils.parallelMapToDouble(X.length, i -> 
            computeDistance(X[i], centers[labels[i]])
        );
        
        return Arrays.stream(distances).sum();
    }
    
    /**
     * Inicializa os centróides usando o método especificado.
     */
    private double[][][] initializeCentroids(double[][][] X, RandomGenerator localRandom) {
        int timeLength = X[0].length;
        int nFeatures = X[0][0].length;
        
        double[][][] localCenters = new double[nClusters][timeLength][nFeatures];
        
        switch (initMethod) {
            case KMEANS_PLUS_PLUS -> initializeKMeansPlusPlus(X, localCenters, localRandom);
            case RANDOM -> initializeRandom(X, localCenters, localRandom);
            case CUSTOM -> throw new UnsupportedOperationException("Inicialização customizada não implementada");
            default -> throw new IllegalArgumentException("Método de inicialização não suportado: " + initMethod);
        }
        
        return localCenters;
    }
    
    /**
     * Inicialização K-means++ paralelizada.
     */
    private void initializeKMeansPlusPlus(double[][][] X, double[][][] centers, RandomGenerator localRandom) {
        int nSamples = X.length;
        
        // Escolher primeiro centróide aleatoriamente
        int firstIndex = localRandom.nextInt(nSamples);
        centers[0] = ArrayUtils.deepCopy2D(X[firstIndex]);
        
        // Escolher centróides restantes
        for (int c = 1; c < nClusters; c++) {
            final int currentC = c; // Variável final para lambda
            
            // Calcular distâncias para centróides existentes (paralelo)
            double[] distances = ParallelUtils.parallelMapToDouble(nSamples, i -> {
                double minDist = Double.POSITIVE_INFINITY;
                for (int j = 0; j < currentC; j++) {
                    double dist = computeDistance(X[i], centers[j]);
                    minDist = Math.min(minDist, dist);
                }
                return minDist * minDist; // Distância ao quadrado
            });
            
            // Escolher próximo centróide baseado nas probabilidades
            double totalDist = Arrays.stream(distances).sum();
            double threshold = localRandom.nextDouble() * totalDist;
            
            double cumSum = 0.0;
            int selectedIndex = 0;
            for (int i = 0; i < nSamples; i++) {
                cumSum += distances[i];
                if (cumSum >= threshold) {
                    selectedIndex = i;
                    break;
                }
            }
            
            centers[c] = ArrayUtils.deepCopy2D(X[selectedIndex]);
        }
    }
    
    /**
     * Inicialização aleatória.
     */
    private void initializeRandom(double[][][] X, double[][][] centers, RandomGenerator localRandom) {
        int nSamples = X.length;
        Set<Integer> selected = new HashSet<>();
        
        for (int c = 0; c < nClusters; c++) {
            int index;
            do {
                index = localRandom.nextInt(nSamples);
            } while (selected.contains(index));
            
            selected.add(index);
            centers[c] = ArrayUtils.deepCopy2D(X[index]);
        }
    }
    
    /**
     * Prediz o cluster para novas amostras usando paralelização.
     */
    public int[] predict(double[][][] X) {
        if (!fitted) {
            throw new IllegalStateException("Modelo deve ser treinado antes da predição");
        }
        
        ParallelUtils.DistanceFunction distanceFunc = this::computeDistance;
        return parallelWorker.assignSamplesToClusters(X, clusterCenters, distanceFunc);
    }
    
    // Getters
    public double[][][] getClusterCenters() { return clusterCenters; }
    public int[] getLabels() { return labels; }
    public double getInertia() { return inertia; }
    public int getNIter() { return nIter; }
    public boolean isFitted() { return fitted; }
    public ParallelUtils.ParallelConfig getParallelConfig() { return parallelConfig; }
    
    /**
     * Libera recursos do pool de threads.
     */
    public void close() {
        if (customPool != null && !customPool.isShutdown()) {
            customPool.shutdown();
        }
    }
}
