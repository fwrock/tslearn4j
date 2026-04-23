package org.tslearn.matrix_profile;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.OptionalInt;
import java.util.concurrent.CompletableFuture;
import java.util.stream.IntStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tslearn.utils.ParallelUtils;

/**
 * Implementação paralelizada do Matrix Profile para análise eficiente de séries temporais.
 * 
 * Esta versão otimizada oferece:
 * - Paralelização automática do algoritmo STAMP
 * - Processamento em lote para grandes datasets
 * - Decomposição em chunks adaptativos
 * - Otimizações específicas para diferentes tamanhos de dados
 * - Computação paralela de motifs e discords
 * - Balanceamento dinâmico de carga
 * 
 * @author TSLearn4J
 */
public class ParallelMatrixProfile extends MatrixProfile {
    
    private static final Logger logger = LoggerFactory.getLogger(ParallelMatrixProfile.class);
    
    // Thresholds para paralelização
    private static final int PARALLEL_THRESHOLD = 1000;
    private static final int CHUNK_SIZE = 200;
    private static final int FFT_PARALLEL_THRESHOLD = 5000;
    
    private final ParallelUtils.ParallelConfig parallelConfig;
    private final boolean enableAdaptiveChunking;
    private final boolean enableFFTParallelization;
    
    /**
     * Builder paralelizado para Matrix Profile.
     */
    public static class ParallelBuilder extends Builder {
        private ParallelUtils.ParallelConfig parallelConfig = ParallelUtils.ParallelConfig.defaultConfig();
        private boolean enableAdaptiveChunking = true;
        private boolean enableFFTParallelization = true;
        
        @Override
        public ParallelBuilder subsequenceLength(int length) {
            super.subsequenceLength(length);
            return this;
        }
        
        @Override
        public ParallelBuilder verbose(boolean verbose) {
            super.verbose(verbose);
            return this;
        }
        
        @Override
        public ParallelBuilder normalize(boolean normalize) {
            super.normalize(normalize);
            return this;
        }
        
        public ParallelBuilder parallelConfig(ParallelUtils.ParallelConfig parallelConfig) {
            this.parallelConfig = parallelConfig;
            return this;
        }
        
        public ParallelBuilder autoConfigureParallelism(int dataSize) {
            this.parallelConfig = ParallelUtils.ParallelConfig.forDataSize(dataSize);
            return this;
        }
        
        public ParallelBuilder enableAdaptiveChunking(boolean enableAdaptiveChunking) {
            this.enableAdaptiveChunking = enableAdaptiveChunking;
            return this;
        }
        
        public ParallelBuilder enableFFTParallelization(boolean enableFFTParallelization) {
            this.enableFFTParallelization = enableFFTParallelization;
            return this;
        }
        
        @Override
        public ParallelMatrixProfile build() {
            return new ParallelMatrixProfile(this);
        }
    }
    
    private ParallelMatrixProfile(ParallelBuilder builder) {
        super(builder);
        this.parallelConfig = builder.parallelConfig;
        this.enableAdaptiveChunking = builder.enableAdaptiveChunking;
        this.enableFFTParallelization = builder.enableFFTParallelization;
    }
    
    /**
     * STAMP paralelizado com otimizações adaptivas.
     */
    @Override
    public MatrixProfileResult stamp(double[] timeSeries) {
        if (timeSeries.length < 2 * getSubsequenceLength()) {
            throw new IllegalArgumentException("Time series too short for given subsequence length");
        }
        
        long startTime = System.currentTimeMillis();
        if (isVerbose()) {
            logger.info("Computing Parallel Matrix Profile for series of length {} with m={}", 
                       timeSeries.length, getSubsequenceLength());
        }
        
        int n = timeSeries.length;
        int m = getSubsequenceLength();
        int profileLength = n - m + 1;
        
        // Decidir estratégia baseada no tamanho
        if (shouldUseParallelization(n, m)) {
            return stampParallel(timeSeries, profileLength);
        } else {
            return super.stamp(timeSeries);
        }
    }
    
    /**
     * Verifica se deve usar paralelização.
     */
    private boolean shouldUseParallelization(int n, int m) {
        return n >= PARALLEL_THRESHOLD && 
               parallelConfig.shouldParallelize(n - m + 1);
    }
    
    /**
     * Implementação paralela do STAMP.
     */
    private MatrixProfileResult stampParallel(double[] timeSeries, int profileLength) {
        int n = timeSeries.length;
        int m = getSubsequenceLength();
        
        // Inicializar estruturas de dados
        double[] matrixProfile = new double[profileLength];
        int[] profileIndex = new int[profileLength];
        double[] leftProfile = new double[profileLength];
        int[] leftProfileIndex = new int[profileLength];
        double[] rightProfile = new double[profileLength];
        int[] rightProfileIndex = new int[profileLength];
        
        Arrays.fill(matrixProfile, Double.POSITIVE_INFINITY);
        Arrays.fill(leftProfile, Double.POSITIVE_INFINITY);
        Arrays.fill(rightProfile, Double.POSITIVE_INFINITY);
        
        // Pré-computar estatísticas necessárias
        double[] means = computeMeansParallel(timeSeries, m);
        double[] stds = computeStandardDeviationsParallel(timeSeries, m, means);
        
        // Escolher estratégia de paralelização
        if (enableAdaptiveChunking && profileLength > CHUNK_SIZE * 4) {
            stampChunkedParallel(timeSeries, matrixProfile, profileIndex, 
                               leftProfile, leftProfileIndex, rightProfile, rightProfileIndex,
                               means, stds);
        } else {
            stampRowParallel(timeSeries, matrixProfile, profileIndex,
                           leftProfile, leftProfileIndex, rightProfile, rightProfileIndex,
                           means, stds);
        }
        
        if (isVerbose()) {
            long duration = System.currentTimeMillis() - System.currentTimeMillis();
            logger.info("Parallel Matrix Profile computation completed in {}ms", duration);
        }
        
        return new MatrixProfileResult(matrixProfile, profileIndex, leftProfile, leftProfileIndex,
                                     rightProfile, rightProfileIndex, m, timeSeries);
    }
    
    /**
     * STAMP com processamento em chunks.
     */
    private void stampChunkedParallel(double[] timeSeries, double[] matrixProfile, int[] profileIndex,
                                    double[] leftProfile, int[] leftProfileIndex,
                                    double[] rightProfile, int[] rightProfileIndex,
                                    double[] means, double[] stds) {
        
        int profileLength = matrixProfile.length;
        int chunkSize = ParallelUtils.getOptimalChunkSize(profileLength, parallelConfig.getParallelism());
        int numChunks = (profileLength + chunkSize - 1) / chunkSize;
        
        // Processar chunks em paralelo
        List<CompletableFuture<Void>> futures = new ArrayList<>();
        
        for (int chunkIndex = 0; chunkIndex < numChunks; chunkIndex++) {
            final int startIdx = chunkIndex * chunkSize;
            final int endIdx = Math.min(startIdx + chunkSize, profileLength);
            
            CompletableFuture<Void> future = CompletableFuture.runAsync(() -> {
                processChunk(timeSeries, startIdx, endIdx, matrixProfile, profileIndex,
                           leftProfile, leftProfileIndex, rightProfile, rightProfileIndex,
                           means, stds);
            }, ParallelUtils.getGlobalPool());
            
            futures.add(future);
        }
        
        // Aguardar conclusão de todos os chunks
        CompletableFuture.allOf(futures.toArray(new CompletableFuture[0])).join();
    }
    
    /**
     * Processa um chunk específico do Matrix Profile.
     */
    private void processChunk(double[] timeSeries, int startIdx, int endIdx,
                            double[] matrixProfile, int[] profileIndex,
                            double[] leftProfile, int[] leftProfileIndex,
                            double[] rightProfile, int[] rightProfileIndex,
                            double[] means, double[] stds) {
        
        int m = getSubsequenceLength();
        int exclusionZone = getExclusionZone();
        
        for (int i = startIdx; i < endIdx; i++) {
            // Extrair subsequência query
            double[] query = Arrays.copyOfRange(timeSeries, i, i + m);
            
            // Computar distâncias para todas as outras subsequências
            double queryMean = means[i];
            double queryStd = stds[i];
            
            if (queryStd == 0) continue; // Skip constant subsequences
            
            // Paralelizar cálculo de distâncias para esta query
            final int qi = i;
            IntStream.range(0, matrixProfile.length)
                    .parallel()
                    .forEach(j -> {
                        if (Math.abs(qi - j) >= exclusionZone) {
                            double distance = computeZNormalizedDistance(timeSeries, qi, j, m, 
                                                                       queryMean, queryStd, means[j], stds[j]);
                            
                            updateProfiles(qi, j, distance, matrixProfile, profileIndex,
                                         leftProfile, leftProfileIndex, rightProfile, rightProfileIndex);
                        }
                    });
        }
    }
    
    /**
     * STAMP com paralelização por linha.
     */
    private void stampRowParallel(double[] timeSeries, double[] matrixProfile, int[] profileIndex,
                                double[] leftProfile, int[] leftProfileIndex,
                                double[] rightProfile, int[] rightProfileIndex,
                                double[] means, double[] stds) {
        
        int m = getSubsequenceLength();
        int exclusionZone = getExclusionZone();
        Object lock = new Object();
        
        // Paralelizar por subsequência query
        IntStream.range(0, matrixProfile.length)
                .parallel()
                .forEach(i -> {
                    double queryMean = means[i];
                    double queryStd = stds[i];
                    
                    if (queryStd == 0) return; // Skip constant subsequences
                    
                    double minDistance = Double.POSITIVE_INFINITY;
                    int minIndex = -1;
                    double minLeftDistance = Double.POSITIVE_INFINITY;
                    int minLeftIndex = -1;
                    double minRightDistance = Double.POSITIVE_INFINITY;
                    int minRightIndex = -1;
                    
                    // Encontrar vizinho mais próximo
                    for (int j = 0; j < matrixProfile.length; j++) {
                        if (Math.abs(i - j) >= exclusionZone) {
                            double distance = computeZNormalizedDistance(timeSeries, i, j, m,
                                                                       queryMean, queryStd, means[j], stds[j]);
                            
                            if (distance < minDistance) {
                                minDistance = distance;
                                minIndex = j;
                            }
                            
                            if (j < i && distance < minLeftDistance) {
                                minLeftDistance = distance;
                                minLeftIndex = j;
                            }
                            
                            if (j > i && distance < minRightDistance) {
                                minRightDistance = distance;
                                minRightIndex = j;
                            }
                        }
                    }
                    
                    // Atualizar profiles (thread-safe)
                    synchronized (lock) {
                        if (minDistance < matrixProfile[i]) {
                            matrixProfile[i] = minDistance;
                            profileIndex[i] = minIndex;
                        }
                        
                        if (minLeftDistance < leftProfile[i]) {
                            leftProfile[i] = minLeftDistance;
                            leftProfileIndex[i] = minLeftIndex;
                        }
                        
                        if (minRightDistance < rightProfile[i]) {
                            rightProfile[i] = minRightDistance;
                            rightProfileIndex[i] = minRightIndex;
                        }
                    }
                });
    }
    
    /**
     * Calcula médias das subsequências em paralelo.
     */
    private double[] computeMeansParallel(double[] timeSeries, int m) {
        int profileLength = timeSeries.length - m + 1;
        
        return ParallelUtils.parallelMapToDouble(profileLength, i -> {
            double sum = 0.0;
            for (int j = 0; j < m; j++) {
                sum += timeSeries[i + j];
            }
            return sum / m;
        });
    }
    
    /**
     * Calcula desvios padrão das subsequências em paralelo.
     */
    private double[] computeStandardDeviationsParallel(double[] timeSeries, int m, double[] means) {
        int profileLength = timeSeries.length - m + 1;
        
        return ParallelUtils.parallelMapToDouble(profileLength, i -> {
            double sumSquares = 0.0;
            double mean = means[i];
            
            for (int j = 0; j < m; j++) {
                double diff = timeSeries[i + j] - mean;
                sumSquares += diff * diff;
            }
            
            return Math.sqrt(sumSquares / m);
        });
    }
    
    /**
     * Computa distância z-normalizada entre duas subsequências.
     */
    private double computeZNormalizedDistance(double[] timeSeries, int i, int j, int m,
                                            double meanI, double stdI, double meanJ, double stdJ) {
        
        if (stdI == 0 || stdJ == 0) {
            return Double.POSITIVE_INFINITY;
        }
        
        double sum = 0.0;
        for (int k = 0; k < m; k++) {
            double zi = (timeSeries[i + k] - meanI) / stdI;
            double zj = (timeSeries[j + k] - meanJ) / stdJ;
            double diff = zi - zj;
            sum += diff * diff;
        }
        
        return Math.sqrt(sum);
    }
    
    /**
     * Atualiza os profiles de forma thread-safe.
     */
    private void updateProfiles(int i, int j, double distance,
                              double[] matrixProfile, int[] profileIndex,
                              double[] leftProfile, int[] leftProfileIndex,
                              double[] rightProfile, int[] rightProfileIndex) {
        
        // Atualizar matrix profile para posição i
        synchronized (matrixProfile) {
            if (distance < matrixProfile[i]) {
                matrixProfile[i] = distance;
                profileIndex[i] = j;
            }
        }
        
        // Atualizar matrix profile para posição j
        synchronized (matrixProfile) {
            if (distance < matrixProfile[j]) {
                matrixProfile[j] = distance;
                profileIndex[j] = i;
            }
        }
        
        // Atualizar left/right profiles
        if (j < i) {
            synchronized (leftProfile) {
                if (distance < leftProfile[i]) {
                    leftProfile[i] = distance;
                    leftProfileIndex[i] = j;
                }
            }
            synchronized (rightProfile) {
                if (distance < rightProfile[j]) {
                    rightProfile[j] = distance;
                    rightProfileIndex[j] = i;
                }
            }
        } else if (j > i) {
            synchronized (rightProfile) {
                if (distance < rightProfile[i]) {
                    rightProfile[i] = distance;
                    rightProfileIndex[i] = j;
                }
            }
            synchronized (leftProfile) {
                if (distance < leftProfile[j]) {
                    leftProfile[j] = distance;
                    leftProfileIndex[j] = i;
                }
            }
        }
    }
    
    /**
     * Descoberta de motifs paralelizada.
     */
    @Override
    public MotifResult findMotifs(double[] timeSeries, int numMotifs) {
        MatrixProfileResult mpResult = stamp(timeSeries);
        double[] matrixProfile = mpResult.getMatrixProfile();
        int[] profileIndex = mpResult.getProfileIndex();
        
        List<MotifResult.MotifPair> motifs = new ArrayList<>();
        boolean[] used = new boolean[matrixProfile.length];
        int exclusionZone = getExclusionZone();
        
        for (int motifCount = 0; motifCount < numMotifs; motifCount++) {
            // Encontrar mínimo não usado em paralelo
            OptionalInt minIdx = IntStream.range(0, matrixProfile.length)
                    .parallel()
                    .filter(i -> !used[i])
                    .reduce((i, j) -> matrixProfile[i] <= matrixProfile[j] ? i : j);
            
            if (!minIdx.isPresent()) break;
            
            int motifIdx = minIdx.getAsInt();
            int matchIdx = profileIndex[motifIdx];
            double distance = matrixProfile[motifIdx];
            
            motifs.add(new MotifResult.MotifPair(motifIdx, matchIdx, distance));
            
            // Marcar zona de exclusão em paralelo
            IntStream.range(0, matrixProfile.length)
                    .parallel()
                    .forEach(i -> {
                        if (Math.abs(i - motifIdx) < exclusionZone || Math.abs(i - matchIdx) < exclusionZone) {
                            used[i] = true;
                        }
                    });
        }
        
        return new MotifResult(motifs, matrixProfile);
    }
    
    /**
     * Descoberta de discords paralelizada.
     */
    @Override
    public DiscordResult findDiscords(double[] timeSeries, int numDiscords) {
        MatrixProfileResult mpResult = stamp(timeSeries);
        double[] matrixProfile = mpResult.getMatrixProfile();
        
        List<DiscordResult.Discord> discords = new ArrayList<>();
        boolean[] used = new boolean[matrixProfile.length];
        int exclusionZone = getExclusionZone();
        
        for (int discordCount = 0; discordCount < numDiscords; discordCount++) {
            // Encontrar máximo não usado em paralelo
            OptionalInt maxIdx = IntStream.range(0, matrixProfile.length)
                    .parallel()
                    .filter(i -> !used[i] && !Double.isInfinite(matrixProfile[i]))
                    .reduce((i, j) -> matrixProfile[i] >= matrixProfile[j] ? i : j);
            
            if (!maxIdx.isPresent()) break;
            
            int discordIdx = maxIdx.getAsInt();
            double distance = matrixProfile[discordIdx];
            
            discords.add(new DiscordResult.Discord(discordIdx, distance));
            
            // Marcar zona de exclusão em paralelo
            IntStream.range(0, matrixProfile.length)
                    .parallel()
                    .forEach(i -> {
                        if (Math.abs(i - discordIdx) < exclusionZone) {
                            used[i] = true;
                        }
                    });
        }
        
        return new DiscordResult(discords, matrixProfile);
    }
    
    /**
     * AB-join paralelizado para duas séries temporais diferentes.
     */
    public MatrixProfileResult abJoin(double[] timeSeriesA, double[] timeSeriesB) {
        if (timeSeriesA.length < getSubsequenceLength() || timeSeriesB.length < getSubsequenceLength()) {
            throw new IllegalArgumentException("Time series too short for given subsequence length");
        }
        
        int m = getSubsequenceLength();
        int profileLengthA = timeSeriesA.length - m + 1;
        int profileLengthB = timeSeriesB.length - m + 1;
        
        double[] matrixProfileA = new double[profileLengthA];
        int[] profileIndexA = new int[profileLengthA];
        Arrays.fill(matrixProfileA, Double.POSITIVE_INFINITY);
        
        // Pré-computar estatísticas
        double[] meansA = computeMeansParallel(timeSeriesA, m);
        double[] stdsA = computeStandardDeviationsParallel(timeSeriesA, m, meansA);
        double[] meansB = computeMeansParallel(timeSeriesB, m);
        double[] stdsB = computeStandardDeviationsParallel(timeSeriesB, m, meansB);
        
        // Computar AB-join em paralelo
        IntStream.range(0, profileLengthA)
                .parallel()
                .forEach(i -> {
                    double queryMeanA = meansA[i];
                    double queryStdA = stdsA[i];
                    
                    if (queryStdA == 0) return;
                    
                    double minDistance = Double.POSITIVE_INFINITY;
                    int minIndex = -1;
                    
                    for (int j = 0; j < profileLengthB; j++) {
                        double distance = computeZNormalizedDistance(timeSeriesA, timeSeriesB, 
                                                                   i, j, m, queryMeanA, queryStdA, meansB[j], stdsB[j]);
                        
                        if (distance < minDistance) {
                            minDistance = distance;
                            minIndex = j;
                        }
                    }
                    
                    matrixProfileA[i] = minDistance;
                    profileIndexA[i] = minIndex;
                });
        
        return new MatrixProfileResult(matrixProfileA, profileIndexA, null, null, null, null, m, timeSeriesA);
    }
    
    /**
     * Computa distância z-normalizada entre subsequências de séries diferentes.
     */
    private double computeZNormalizedDistance(double[] seriesA, double[] seriesB, int i, int j, int m,
                                            double meanA, double stdA, double meanB, double stdB) {
        
        if (stdA == 0 || stdB == 0) {
            return Double.POSITIVE_INFINITY;
        }
        
        double sum = 0.0;
        for (int k = 0; k < m; k++) {
            double zA = (seriesA[i + k] - meanA) / stdA;
            double zB = (seriesB[j + k] - meanB) / stdB;
            double diff = zA - zB;
            sum += diff * diff;
        }
        
        return Math.sqrt(sum);
    }
    
    /**
     * Processamento em lote para múltiplas séries temporais.
     */
    public MatrixProfileResult[] stampBatch(double[][] timeSeriesArray) {
        return ParallelUtils.parallelMap(timeSeriesArray.length,
            i -> stamp(timeSeriesArray[i]),
            MatrixProfileResult[]::new);
    }
    
    // Métodos auxiliares para acessar propriedades da classe pai
    private int getExclusionZone() {
        return getSubsequenceLength() / 2;
    }
    
    // Getters para configuração
    public ParallelUtils.ParallelConfig getParallelConfig() {
        return parallelConfig;
    }
    
    public boolean isAdaptiveChunkingEnabled() {
        return enableAdaptiveChunking;
    }
    
    public boolean isFFTParallelizationEnabled() {
        return enableFFTParallelization;
    }
}
