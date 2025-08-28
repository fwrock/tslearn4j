package org.tslearn.clustering;

import java.util.*;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.random.Well19937c;
import org.apache.commons.math3.util.FastMath;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tslearn.barycenters.DTWBarycenter;
import org.tslearn.barycenters.EuclideanBarycenter;
import org.tslearn.metrics.DTW;
import org.tslearn.utils.ArrayUtils;

/**
 * K-means clustering para dados de séries temporais.
 * 
 * Implementação Java equivalente ao TimeSeriesKMeans do Python tslearn,
 * suportando múltiplas métricas de distância incluindo DTW.
 * 
 * @author TSLearn4J
 */
public class TimeSeriesKMeans {
    
    private static final Logger logger = LoggerFactory.getLogger(TimeSeriesKMeans.class);
    
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
    
    // Parâmetros específicos das métricas
    private final Map<String, Object> metricParams;
    
    // Resultados do treinamento
    private double[][][] clusterCenters;
    private int[] labels;
    private double inertia;
    private int nIter;
    private boolean fitted = false;
    
    // DTW instance para reutilização
    private final DTW dtw;
    
    /**
     * Construtor principal do TimeSeriesKMeans.
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
        
        public Builder metricParam(String key, Object value) {
            this.metricParams.put(key, value);
            return this;
        }
        
        public TimeSeriesKMeans build() {
            return new TimeSeriesKMeans(this);
        }
    }
    
    private TimeSeriesKMeans(Builder builder) {
        this.nClusters = builder.nClusters;
        this.maxIter = builder.maxIter;
        this.tolerance = builder.tolerance;
        this.nInit = builder.nInit;
        this.metric = builder.metric;
        this.maxIterBarycenter = builder.maxIterBarycenter;
        this.initMethod = builder.initMethod;
        this.verbose = builder.verbose;
        this.dtwInertia = builder.dtwInertia;
        this.metricParams = new HashMap<>(builder.metricParams);
        
        // Inicializar gerador aleatório
        if (builder.randomSeed != null) {
            this.random = new Well19937c(builder.randomSeed);
        } else {
            this.random = new Well19937c();
        }
        
        // Inicializar DTW com parâmetros
        if (metricParams.containsKey("sakoeChiba")) {
            int sakoeChiba = (Integer) metricParams.get("sakoeChiba");
            this.dtw = new DTW.Builder()
                    .sakoeChibaRadius(sakoeChiba)
                    .build();
        } else {
            this.dtw = new DTW.Builder().build();
        }
        
        if (verbose) {
            logger.info("TimeSeriesKMeans inicializado: {} clusters, métrica {}", 
                       nClusters, metric);
        }
    }
    
    /**
     * Treina o modelo de clustering com o dataset fornecido.
     */
    public TimeSeriesKMeans fit(double[][][] X) {
        if (X == null || X.length == 0) {
            throw new IllegalArgumentException("Dataset não pode ser vazio");
        }
        
        int nSamples = X.length;
        int timeLength = X[0].length;
        int nFeatures = X[0][0].length;
        
        if (verbose) {
            logger.info("Treinando TimeSeriesKMeans: {} amostras, {} timesteps, {} features", 
                       nSamples, timeLength, nFeatures);
        }
        
        // Validar entrada
        validateInput(X);
        
        double bestInertia = Double.POSITIVE_INFINITY;
        double[][][] bestCenters = null;
        int bestNIter = 0;
        int[] bestLabels = null;
        
        // Múltiplas inicializações
        for (int init = 0; init < nInit; init++) {
            if (verbose && nInit > 1) {
                logger.info("Inicialização {} de {}", init + 1, nInit);
            }
            
            try {
                // Executar uma inicialização
                fitOneInit(X);
                
                // Manter a melhor solução
                if (inertia < bestInertia) {
                    bestInertia = inertia;
                    bestCenters = ArrayUtils.deepCopy3D(clusterCenters);
                    bestNIter = nIter;
                    bestLabels = labels.clone();
                    
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
        
        // Definir os melhores resultados
        this.clusterCenters = bestCenters;
        this.inertia = bestInertia;
        this.nIter = bestNIter;
        this.labels = bestLabels;
        this.fitted = true;
        
        if (verbose) {
            logger.info("Treinamento concluído: inércia final = {:.6f}, {} iterações", 
                       inertia, nIter);
        }
        
        return this;
    }
    
    /**
     * Executa uma única inicialização do algoritmo K-means.
     */
    private void fitOneInit(double[][][] X) {
        int nSamples = X.length;
        int timeLength = X[0].length;
        int nFeatures = X[0][0].length;
        
        // Inicializar centróides
        initializeCentroids(X);
        
        double oldInertia = Double.POSITIVE_INFINITY;
        int iter;
        
        for (iter = 0; iter < maxIter; iter++) {
            // Atribuir amostras aos clusters
            assignSamplesToClusters(X);
            
            if (verbose) {
                System.out.printf("%.6f --> ", inertia);
            }
            
            // Atualizar centróides
            updateCentroids(X);
            
            // Verificar convergência
            if (FastMath.abs(oldInertia - inertia) < tolerance) {
                if (verbose) {
                    System.out.println("Convergiu");
                }
                break;
            }
            oldInertia = inertia;
        }
        
        if (verbose) {
            System.out.println();
        }
        
        this.nIter = iter + 1;
    }
    
    /**
     * Inicializa os centróides usando o método especificado.
     */
    private void initializeCentroids(double[][][] X) {
        int nSamples = X.length;
        int timeLength = X[0].length;
        int nFeatures = X[0][0].length;
        
        this.clusterCenters = new double[nClusters][timeLength][nFeatures];
        
        switch (initMethod) {
            case KMEANS_PLUS_PLUS:
                initializeKMeansPlusPlus(X);
                break;
            case RANDOM:
                initializeRandom(X);
                break;
            default:
                throw new IllegalArgumentException("Método de inicialização não suportado: " + initMethod);
        }
        
        if (verbose) {
            logger.debug("Centróides inicializados usando {}", initMethod);
        }
    }
    
    /**
     * Inicialização K-means++ adaptada para séries temporais.
     */
    private void initializeKMeansPlusPlus(double[][][] X) {
        int nSamples = X.length;
        
        // Escolher primeiro centróide aleatoriamente
        int firstIndex = random.nextInt(nSamples);
        clusterCenters[0] = ArrayUtils.deepCopy2D(X[firstIndex]);
        
        // Escolher centróides restantes
        for (int c = 1; c < nClusters; c++) {
            double[] distances = new double[nSamples];
            double totalDistance = 0.0;
            
            // Calcular distâncias para o centróide mais próximo
            for (int i = 0; i < nSamples; i++) {
                double minDist = Double.POSITIVE_INFINITY;
                for (int k = 0; k < c; k++) {
                    double dist = computeDistance(X[i], clusterCenters[k]);
                    minDist = Math.min(minDist, dist);
                }
                distances[i] = minDist * minDist; // Distância ao quadrado
                totalDistance += distances[i];
            }
            
            // Escolher próximo centróide proporcionalmente à distância
            double threshold = random.nextDouble() * totalDistance;
            double cumSum = 0.0;
            int selectedIndex = 0;
            
            for (int i = 0; i < nSamples; i++) {
                cumSum += distances[i];
                if (cumSum >= threshold) {
                    selectedIndex = i;
                    break;
                }
            }
            
            clusterCenters[c] = ArrayUtils.deepCopy2D(X[selectedIndex]);
        }
    }
    
    /**
     * Inicialização aleatória simples.
     */
    private void initializeRandom(double[][][] X) {
        int nSamples = X.length;
        Set<Integer> selectedIndices = new HashSet<>();
        
        for (int c = 0; c < nClusters; c++) {
            int index;
            do {
                index = random.nextInt(nSamples);
            } while (selectedIndices.contains(index));
            
            selectedIndices.add(index);
            clusterCenters[c] = ArrayUtils.deepCopy2D(X[index]);
        }
    }
    
    /**
     * Atribui cada amostra ao cluster mais próximo.
     */
    private void assignSamplesToClusters(double[][][] X) {
        int nSamples = X.length;
        this.labels = new int[nSamples];
        
        double totalInertia = 0.0;
        int[] clusterSizes = new int[nClusters];
        
        for (int i = 0; i < nSamples; i++) {
            double minDistance = Double.POSITIVE_INFINITY;
            int bestCluster = 0;
            
            for (int k = 0; k < nClusters; k++) {
                double distance = computeDistance(X[i], clusterCenters[k]);
                if (distance < minDistance) {
                    minDistance = distance;
                    bestCluster = k;
                }
            }
            
            labels[i] = bestCluster;
            clusterSizes[bestCluster]++;
            
            // Calcular inércia (pode usar DTW se especificado)
            if (dtwInertia && metric != Metric.DTW) {
                totalInertia += dtw.distance(X[i], clusterCenters[bestCluster]);
            } else {
                totalInertia += minDistance;
            }
        }
        
        // Verificar clusters vazios
        for (int k = 0; k < nClusters; k++) {
            if (clusterSizes[k] == 0) {
                throw new RuntimeException("Cluster vazio detectado: " + k);
            }
        }
        
        this.inertia = totalInertia;
    }
    
    /**
     * Atualiza os centróides baseado nas atribuições atuais.
     */
    private void updateCentroids(double[][][] X) {
        int nSamples = X.length;
        
        for (int k = 0; k < nClusters; k++) {
            // Coletar amostras do cluster k
            List<double[][]> clusterSamples = new ArrayList<>();
            for (int i = 0; i < nSamples; i++) {
                if (labels[i] == k) {
                    clusterSamples.add(X[i]);
                }
            }
            
            if (clusterSamples.isEmpty()) {
                continue; // Cluster vazio, manter centróide atual
            }
            
            // Calcular novo centróide baseado na métrica
            switch (metric) {
                case EUCLIDEAN:
                    clusterCenters[k] = EuclideanBarycenter.compute(
                        clusterSamples.toArray(new double[0][][])
                    );
                    break;
                case DTW:
                    clusterCenters[k] = DTWBarycenter.compute(
                        clusterSamples.toArray(new double[0][][]),
                        clusterCenters[k], // Usar centróide atual como inicialização
                        maxIterBarycenter
                    );
                    break;
                case SOFT_DTW:
                    // TODO: Implementar SoftDTW barycenter
                    throw new UnsupportedOperationException("SoftDTW não implementado ainda");
                default:
                    throw new IllegalArgumentException("Métrica não suportada: " + metric);
            }
        }
    }
    
    /**
     * Calcula a distância entre duas séries temporais baseada na métrica configurada.
     */
    private double computeDistance(double[][] ts1, double[][] ts2) {
        switch (metric) {
            case EUCLIDEAN:
                return computeEuclideanDistance(ts1, ts2);
            case DTW:
                return dtw.distance(ts1, ts2);
            case SOFT_DTW:
                // TODO: Implementar SoftDTW
                throw new UnsupportedOperationException("SoftDTW não implementado ainda");
            default:
                throw new IllegalArgumentException("Métrica não suportada: " + metric);
        }
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
     * Prediz o cluster para novas amostras.
     */
    public int[] predict(double[][][] X) {
        if (!fitted) {
            throw new IllegalStateException("Modelo deve ser treinado antes da predição");
        }
        
        int nSamples = X.length;
        int[] predictions = new int[nSamples];
        
        for (int i = 0; i < nSamples; i++) {
            double minDistance = Double.POSITIVE_INFINITY;
            int bestCluster = 0;
            
            for (int k = 0; k < nClusters; k++) {
                double distance = computeDistance(X[i], clusterCenters[k]);
                if (distance < minDistance) {
                    minDistance = distance;
                    bestCluster = k;
                }
            }
            
            predictions[i] = bestCluster;
        }
        
        return predictions;
    }
    
    /**
     * Treina o modelo e retorna as predições.
     */
    public int[] fitPredict(double[][][] X) {
        fit(X);
        return labels.clone();
    }
    
    /**
     * Calcula a matriz de distâncias para os centróides.
     */
    public double[][] transform(double[][][] X) {
        if (!fitted) {
            throw new IllegalStateException("Modelo deve ser treinado antes da transformação");
        }
        
        int nSamples = X.length;
        double[][] distances = new double[nSamples][nClusters];
        
        for (int i = 0; i < nSamples; i++) {
            for (int k = 0; k < nClusters; k++) {
                distances[i][k] = computeDistance(X[i], clusterCenters[k]);
            }
        }
        
        return distances;
    }
    
    /**
     * Valida o dataset de entrada.
     */
    private void validateInput(double[][][] X) {
        if (X.length < nClusters) {
            throw new IllegalArgumentException(
                "Número de amostras deve ser >= número de clusters");
        }
        
        int expectedTimeLength = X[0].length;
        int expectedFeatures = X[0][0].length;
        
        for (int i = 0; i < X.length; i++) {
            if (X[i].length != expectedTimeLength) {
                throw new IllegalArgumentException(
                    "Todas as séries devem ter o mesmo comprimento temporal");
            }
            for (int t = 0; t < X[i].length; t++) {
                if (X[i][t].length != expectedFeatures) {
                    throw new IllegalArgumentException(
                        "Todas as séries devem ter o mesmo número de features");
                }
            }
        }
    }
    
    // Getters
    public double[][][] getClusterCenters() {
        if (!fitted) return null;
        return ArrayUtils.deepCopy3D(clusterCenters);
    }
    
    public int[] getLabels() {
        if (!fitted) return null;
        return labels.clone();
    }
    
    public double getInertia() {
        return inertia;
    }
    
    public int getNumIterations() {
        return nIter;
    }
    
    public boolean isFitted() {
        return fitted;
    }
    
    public Metric getMetric() {
        return metric;
    }
    
    public int getNumClusters() {
        return nClusters;
    }
}
