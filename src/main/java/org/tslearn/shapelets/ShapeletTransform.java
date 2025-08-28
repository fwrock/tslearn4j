package org.tslearn.shapelets;

import java.util.*;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.random.Well19937c;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tslearn.clustering.TimeSeriesKMeans;

/**
 * Transformação Shapelet para séries temporais.
 * 
 * Esta implementação extrai shapelets candidatos de um dataset de treinamento
 * e os utiliza para transformar séries temporais em um espaço de features
 * baseado nas distâncias aos shapelets mais discriminativos.
 * 
 * Baseado no trabalho de Ye & Keogh (2009) e Grabocka et al. (2014).
 * 
 * @author TSLearn4J
 */
public class ShapeletTransform {
    
    private static final Logger logger = LoggerFactory.getLogger(ShapeletTransform.class);
    
    public enum ShapeletSelectionMethod {
        INFORMATION_GAIN,
        F_STATISTIC,
        MOODS_MEDIAN,
        KRUSKAL_WALLIS
    }
    
    public enum InitializationMethod {
        RANDOM,
        KMEANS,
        CLASS_BALANCED
    }
    
    // Parâmetros de configuração
    private final int minShapeletLength;
    private final int maxShapeletLength;
    private final int numShapelets;
    private final int maxCandidates;
    private final ShapeletSelectionMethod selectionMethod;
    private final InitializationMethod initMethod;
    private final boolean removeSimilar;
    private final double similarityThreshold;
    private final boolean verbose;
    private final RandomGenerator random;
    
    // Estado do modelo
    private List<Shapelet> shapelets;
    private boolean fitted = false;
    private Map<String, Integer> classToIndex;
    private String[] indexToClass;
    
    /**
     * Builder para ShapeletTransform.
     */
    public static class Builder {
        private int minShapeletLength = 3;
        private int maxShapeletLength = -1; // Será definido automaticamente
        private int numShapelets = 100;
        private int maxCandidates = 10000;
        private ShapeletSelectionMethod selectionMethod = ShapeletSelectionMethod.INFORMATION_GAIN;
        private InitializationMethod initMethod = InitializationMethod.RANDOM;
        private boolean removeSimilar = true;
        private double similarityThreshold = 0.1;
        private boolean verbose = false;
        private Long randomSeed = null;
        
        public Builder minShapeletLength(int length) {
            this.minShapeletLength = length;
            return this;
        }
        
        public Builder maxShapeletLength(int length) {
            this.maxShapeletLength = length;
            return this;
        }
        
        public Builder numShapelets(int num) {
            this.numShapelets = num;
            return this;
        }
        
        public Builder maxCandidates(int max) {
            this.maxCandidates = max;
            return this;
        }
        
        public Builder selectionMethod(ShapeletSelectionMethod method) {
            this.selectionMethod = method;
            return this;
        }
        
        public Builder initializationMethod(InitializationMethod method) {
            this.initMethod = method;
            return this;
        }
        
        public Builder removeSimilar(boolean remove) {
            this.removeSimilar = remove;
            return this;
        }
        
        public Builder similarityThreshold(double threshold) {
            this.similarityThreshold = threshold;
            return this;
        }
        
        public Builder verbose(boolean verbose) {
            this.verbose = verbose;
            return this;
        }
        
        public Builder randomSeed(long seed) {
            this.randomSeed = seed;
            return this;
        }
        
        public ShapeletTransform build() {
            return new ShapeletTransform(this);
        }
    }
    
    private ShapeletTransform(Builder builder) {
        this.minShapeletLength = builder.minShapeletLength;
        this.maxShapeletLength = builder.maxShapeletLength;
        this.numShapelets = builder.numShapelets;
        this.maxCandidates = builder.maxCandidates;
        this.selectionMethod = builder.selectionMethod;
        this.initMethod = builder.initMethod;
        this.removeSimilar = builder.removeSimilar;
        this.similarityThreshold = builder.similarityThreshold;
        this.verbose = builder.verbose;
        
        if (builder.randomSeed != null) {
            this.random = new Well19937c(builder.randomSeed);
        } else {
            this.random = new Well19937c();
        }
        
        if (verbose) {
            logger.info("ShapeletTransform configurado: {} shapelets, comprimento [{}, {}]", 
                       numShapelets, minShapeletLength, maxShapeletLength);
        }
    }
    
    /**
     * Treina o transformador com dados rotulados.
     * 
     * @param X Dataset de séries temporais [n_samples][time_length][n_features]
     * @param y Labels das classes [n_samples]
     * @return Esta instância (para method chaining)
     */
    public ShapeletTransform fit(double[][][] X, String[] y) {
        if (X == null || y == null || X.length != y.length) {
            throw new IllegalArgumentException("X e y devem ter o mesmo número de amostras");
        }
        
        if (verbose) {
            logger.info("Iniciando treinamento com {} séries temporais", X.length);
        }
        
        // Configurar mapeamento de classes
        setupClassMapping(y);
        
        // Determinar comprimento máximo se não especificado
        int actualMaxLength = maxShapeletLength;
        if (actualMaxLength <= 0) {
            actualMaxLength = Math.min(X[0].length / 2, 50); // Heurística
        }
        
        // Gerar candidatos a shapelets
        List<Shapelet> candidates = generateShapeletCandidates(X, y, actualMaxLength);
        
        if (verbose) {
            logger.info("Gerados {} candidatos a shapelets", candidates.size());
        }
        
        // Avaliar qualidade dos candidatos
        evaluateShapeletQuality(candidates, X, y);
        
        // Selecionar melhores shapelets
        this.shapelets = selectBestShapelets(candidates);
        
        // Remover shapelets similares se configurado
        if (removeSimilar) {
            removeSimilarShapelets();
        }
        
        this.fitted = true;
        
        if (verbose) {
            logger.info("Treinamento concluído. {} shapelets selecionados", shapelets.size());
        }
        
        return this;
    }
    
    /**
     * Transforma séries temporais usando os shapelets aprendidos.
     * 
     * @param X Dataset de séries temporais [n_samples][time_length][n_features]
     * @return Matriz de features transformadas [n_samples][n_shapelets]
     */
    public double[][] transform(double[][][] X) {
        if (!fitted) {
            throw new IllegalStateException("Modelo deve ser treinado antes da transformação");
        }
        
        double[][] transformed = new double[X.length][shapelets.size()];
        
        for (int i = 0; i < X.length; i++) {
            for (int j = 0; j < shapelets.size(); j++) {
                Shapelet.ShapeletMatch match = shapelets.get(j).findBestMatch(X[i]);
                transformed[i][j] = match.getDistance();
            }
        }
        
        return transformed;
    }
    
    /**
     * Treina e transforma em uma única operação.
     */
    public double[][] fitTransform(double[][][] X, String[] y) {
        fit(X, y);
        return transform(X);
    }
    
    /**
     * Gera candidatos a shapelets do dataset de treinamento.
     */
    private List<Shapelet> generateShapeletCandidates(double[][][] X, String[] y, int maxLength) {
        List<Shapelet> candidates = new ArrayList<>();
        
        switch (initMethod) {
            case RANDOM:
                candidates = generateRandomCandidates(X, y, maxLength);
                break;
            case KMEANS:
                candidates = generateKMeansCandidates(X, y, maxLength);
                break;
            case CLASS_BALANCED:
                candidates = generateClassBalancedCandidates(X, y, maxLength);
                break;
        }
        
        return candidates;
    }
    
    /**
     * Gera candidatos aleatórios.
     */
    private List<Shapelet> generateRandomCandidates(double[][][] X, String[] y, int maxLength) {
        List<Shapelet> candidates = new ArrayList<>();
        int candidatesPerLength = maxCandidates / (maxLength - minShapeletLength + 1);
        
        for (int length = minShapeletLength; length <= maxLength; length++) {
            for (int c = 0; c < candidatesPerLength; c++) {
                // Escolher série temporal aleatória
                int seriesIndex = random.nextInt(X.length);
                
                // Escolher posição aleatória
                int maxStart = X[seriesIndex].length - length;
                if (maxStart <= 0) continue;
                
                int startPos = random.nextInt(maxStart + 1);
                
                // Extrair subsequência
                Shapelet candidate = Shapelet.extractSubsequence(X[seriesIndex], startPos, length);
                candidate = new Shapelet(candidate.getValues(), seriesIndex, startPos, 0.0, y[seriesIndex]);
                
                candidates.add(candidate);
            }
        }
        
        return candidates;
    }
    
    /**
     * Gera candidatos usando K-means clustering.
     */
    private List<Shapelet> generateKMeansCandidates(double[][][] X, String[] y, int maxLength) {
        List<Shapelet> candidates = new ArrayList<>();
        
        for (int length = minShapeletLength; length <= maxLength; length++) {
            // Extrair todas as subsequências deste comprimento
            List<double[][]> subsequences = new ArrayList<>();
            List<String> subsequenceLabels = new ArrayList<>();
            List<Integer> sourceIndices = new ArrayList<>();
            List<Integer> startPositions = new ArrayList<>();
            
            for (int i = 0; i < X.length; i++) {
                for (int start = 0; start <= X[i].length - length; start++) {
                    double[][] subseq = new double[length][X[i][0].length];
                    for (int t = 0; t < length; t++) {
                        System.arraycopy(X[i][start + t], 0, subseq[t], 0, X[i][0].length);
                    }
                    subsequences.add(subseq);
                    subsequenceLabels.add(y[i]);
                    sourceIndices.add(i);
                    startPositions.add(start);
                }
            }
            
            // Aplicar K-means
            int nClusters = Math.min(numShapelets / (maxLength - minShapeletLength + 1), 
                                   subsequences.size());
            if (nClusters <= 0) continue;
            
            double[][][] subsequenceArray = subsequences.toArray(new double[0][][]);
            
            TimeSeriesKMeans kmeans = new TimeSeriesKMeans.Builder()
                    .nClusters(nClusters)
                    .metric(TimeSeriesKMeans.Metric.EUCLIDEAN)
                    .maxIter(50)
                    .nInit(3)
                    .randomSeed(random.nextLong())
                    .build();
            
            kmeans.fit(subsequenceArray);
            double[][][] centroids = kmeans.getClusterCenters();
            
            // Converter centroides em shapelets
            for (int c = 0; c < centroids.length; c++) {
                Shapelet candidate = new Shapelet(centroids[c], -1, -1, 0.0, "centroid");
                candidates.add(candidate);
            }
        }
        
        return candidates;
    }
    
    /**
     * Gera candidatos balanceados por classe.
     */
    private List<Shapelet> generateClassBalancedCandidates(double[][][] X, String[] y, int maxLength) {
        List<Shapelet> candidates = new ArrayList<>();
        
        // Agrupar por classe
        Map<String, List<Integer>> classSamples = new HashMap<>();
        for (int i = 0; i < y.length; i++) {
            classSamples.computeIfAbsent(y[i], k -> new ArrayList<>()).add(i);
        }
        
        int candidatesPerClass = maxCandidates / classSamples.size();
        int candidatesPerLength = candidatesPerClass / (maxLength - minShapeletLength + 1);
        
        for (String className : classSamples.keySet()) {
            List<Integer> classIndices = classSamples.get(className);
            
            for (int length = minShapeletLength; length <= maxLength; length++) {
                for (int c = 0; c < candidatesPerLength; c++) {
                    // Escolher série da classe
                    int seriesIndex = classIndices.get(random.nextInt(classIndices.size()));
                    
                    // Escolher posição aleatória
                    int maxStart = X[seriesIndex].length - length;
                    if (maxStart <= 0) continue;
                    
                    int startPos = random.nextInt(maxStart + 1);
                    
                    // Extrair subsequência
                    Shapelet candidate = Shapelet.extractSubsequence(X[seriesIndex], startPos, length);
                    candidate = new Shapelet(candidate.getValues(), seriesIndex, startPos, 
                                           0.0, className);
                    
                    candidates.add(candidate);
                }
            }
        }
        
        return candidates;
    }
    
    /**
     * Avalia a qualidade discriminativa dos candidatos.
     */
    private void evaluateShapeletQuality(List<Shapelet> candidates, double[][][] X, String[] y) {
        if (verbose) {
            logger.info("Avaliando qualidade de {} candidatos...", candidates.size());
        }
        
        for (int i = 0; i < candidates.size(); i++) {
            Shapelet candidate = candidates.get(i);
            double quality = calculateShapeletQuality(candidate, X, y);
            
            // Atualizar qualidade do shapelet
            candidates.set(i, new Shapelet(candidate.getValues(), 
                                         candidate.getSourceIndex(),
                                         candidate.getStartPosition(),
                                         quality, candidate.getLabel()));
            
            if (verbose && (i + 1) % 1000 == 0) {
                logger.debug("Avaliados {}/{} candidatos", i + 1, candidates.size());
            }
        }
    }
    
    /**
     * Calcula a qualidade discriminativa de um shapelet usando Information Gain.
     */
    private double calculateShapeletQuality(Shapelet shapelet, double[][][] X, String[] y) {
        // Calcular distâncias para todas as séries
        double[] distances = new double[X.length];
        for (int i = 0; i < X.length; i++) {
            Shapelet.ShapeletMatch match = shapelet.findBestMatch(X[i]);
            distances[i] = match.getDistance();
        }
        
        switch (selectionMethod) {
            case INFORMATION_GAIN:
                return calculateInformationGain(distances, y);
            case F_STATISTIC:
                return calculateFStatistic(distances, y);
            case MOODS_MEDIAN:
                return calculateMoodsMedian(distances, y);
            case KRUSKAL_WALLIS:
                return calculateKruskalWallis(distances, y);
            default:
                return calculateInformationGain(distances, y);
        }
    }
    
    /**
     * Calcula Information Gain para um shapelet.
     */
    private double calculateInformationGain(double[] distances, String[] y) {
        // Encontrar threshold ótimo
        double[] sortedDistances = distances.clone();
        Arrays.sort(sortedDistances);
        
        double maxGain = Double.NEGATIVE_INFINITY;
        
        for (int i = 1; i < sortedDistances.length; i++) {
            double threshold = (sortedDistances[i-1] + sortedDistances[i]) / 2.0;
            double gain = calculateInformationGainForThreshold(distances, y, threshold);
            maxGain = Math.max(maxGain, gain);
        }
        
        return maxGain;
    }
    
    /**
     * Calcula Information Gain para um threshold específico.
     */
    private double calculateInformationGainForThreshold(double[] distances, String[] y, double threshold) {
        // Calcular entropia original
        double originalEntropy = calculateEntropy(y);
        
        // Dividir dados baseado no threshold
        List<String> leftLabels = new ArrayList<>();
        List<String> rightLabels = new ArrayList<>();
        
        for (int i = 0; i < distances.length; i++) {
            if (distances[i] <= threshold) {
                leftLabels.add(y[i]);
            } else {
                rightLabels.add(y[i]);
            }
        }
        
        if (leftLabels.isEmpty() || rightLabels.isEmpty()) {
            return 0.0;
        }
        
        // Calcular entropia ponderada
        double leftWeight = (double) leftLabels.size() / y.length;
        double rightWeight = (double) rightLabels.size() / y.length;
        
        double leftEntropy = calculateEntropy(leftLabels.toArray(new String[0]));
        double rightEntropy = calculateEntropy(rightLabels.toArray(new String[0]));
        
        double weightedEntropy = leftWeight * leftEntropy + rightWeight * rightEntropy;
        
        return originalEntropy - weightedEntropy;
    }
    
    /**
     * Calcula a entropia de um conjunto de labels.
     */
    private double calculateEntropy(String[] labels) {
        Map<String, Integer> counts = new HashMap<>();
        for (String label : labels) {
            counts.put(label, counts.getOrDefault(label, 0) + 1);
        }
        
        double entropy = 0.0;
        for (int count : counts.values()) {
            double probability = (double) count / labels.length;
            if (probability > 0) {
                entropy -= probability * Math.log(probability) / Math.log(2);
            }
        }
        
        return entropy;
    }
    
    /**
     * Implementação simplificada do F-statistic.
     */
    private double calculateFStatistic(double[] distances, String[] y) {
        // Agrupar distâncias por classe
        Map<String, List<Double>> classDistances = new HashMap<>();
        for (int i = 0; i < y.length; i++) {
            classDistances.computeIfAbsent(y[i], k -> new ArrayList<>()).add(distances[i]);
        }
        
        if (classDistances.size() < 2) return 0.0;
        
        // Calcular F-statistic simplificado
        double overallMean = Arrays.stream(distances).average().orElse(0.0);
        double betweenClassVariance = 0.0;
        double withinClassVariance = 0.0;
        
        for (List<Double> classData : classDistances.values()) {
            double classMean = classData.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
            betweenClassVariance += classData.size() * Math.pow(classMean - overallMean, 2);
            
            for (double value : classData) {
                withinClassVariance += Math.pow(value - classMean, 2);
            }
        }
        
        betweenClassVariance /= (classDistances.size() - 1);
        withinClassVariance /= (distances.length - classDistances.size());
        
        return withinClassVariance > 0 ? betweenClassVariance / withinClassVariance : 0.0;
    }
    
    /**
     * Implementação simplificada do Mood's Median Test.
     */
    private double calculateMoodsMedian(double[] distances, String[] y) {
        // Implementação simplificada - usar diferença de medianas
        Map<String, List<Double>> classDistances = new HashMap<>();
        for (int i = 0; i < y.length; i++) {
            classDistances.computeIfAbsent(y[i], k -> new ArrayList<>()).add(distances[i]);
        }
        
        if (classDistances.size() < 2) return 0.0;
        
        List<Double> medians = new ArrayList<>();
        for (List<Double> classData : classDistances.values()) {
            Collections.sort(classData);
            double median = classData.get(classData.size() / 2);
            medians.add(median);
        }
        
        double maxMedian = Collections.max(medians);
        double minMedian = Collections.min(medians);
        
        return maxMedian - minMedian;
    }
    
    /**
     * Implementação simplificada do Kruskal-Wallis Test.
     */
    private double calculateKruskalWallis(double[] distances, String[] y) {
        // Para simplificar, usar variância entre classes
        return calculateFStatistic(distances, y);
    }
    
    /**
     * Seleciona os melhores shapelets baseado na qualidade.
     */
    private List<Shapelet> selectBestShapelets(List<Shapelet> candidates) {
        // Ordenar por qualidade (descendente)
        candidates.sort((a, b) -> Double.compare(b.getQualityScore(), a.getQualityScore()));
        
        // Selecionar os melhores
        int selectCount = Math.min(numShapelets, candidates.size());
        return new ArrayList<>(candidates.subList(0, selectCount));
    }
    
    /**
     * Remove shapelets similares.
     */
    private void removeSimilarShapelets() {
        List<Shapelet> filtered = new ArrayList<>();
        
        for (Shapelet candidate : shapelets) {
            boolean isSimilar = false;
            
            for (Shapelet existing : filtered) {
                if (candidate.isSimilarTo(existing, similarityThreshold)) {
                    isSimilar = true;
                    break;
                }
            }
            
            if (!isSimilar) {
                filtered.add(candidate);
            }
        }
        
        this.shapelets = filtered;
        
        if (verbose) {
            logger.info("Removidos shapelets similares. Restaram {} únicos", shapelets.size());
        }
    }
    
    /**
     * Configura o mapeamento de classes.
     */
    private void setupClassMapping(String[] y) {
        Set<String> uniqueClasses = new HashSet<>(Arrays.asList(y));
        this.classToIndex = new HashMap<>();
        this.indexToClass = new String[uniqueClasses.size()];
        
        int index = 0;
        for (String className : uniqueClasses) {
            classToIndex.put(className, index);
            indexToClass[index] = className;
            index++;
        }
    }
    
    // Getters
    public List<Shapelet> getShapelets() {
        return fitted ? new ArrayList<>(shapelets) : null;
    }
    
    public boolean isFitted() {
        return fitted;
    }
    
    public int getNumShapelets() {
        return fitted ? shapelets.size() : 0;
    }
    
    public String[] getClasses() {
        return fitted ? indexToClass.clone() : null;
    }
}
