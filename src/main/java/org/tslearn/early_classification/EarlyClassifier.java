package org.tslearn.early_classification;

import java.util.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Early Classification para Séries Temporais usando Reliable Early Classification (RELCLASS).
 * 
 * Esta implementação permite classificar séries temporais antes de observar toda a sequência,
 * baseando-se em confiança estatística e múltiplos classificadores.
 * 
 * Características:
 * - Classificação baseada em confiança
 * - Múltiplos classificadores de base
 * - Estratégias adaptativas de stopping
 * - Suporte a séries multivariadas
 * - Análise de trade-off precisão vs. earliness
 */
public class EarlyClassifier {
    
    private static final Logger logger = LoggerFactory.getLogger(EarlyClassifier.class);
    
    /**
     * Estratégias de stopping para early classification.
     */
    public enum StoppingStrategy {
        /** Para quando confiança excede threshold */
        CONFIDENCE_THRESHOLD,
        /** Para quando margem entre classes é suficiente */
        MARGIN_BASED,
        /** Para baseado em consenso entre classificadores */
        ENSEMBLE_CONSENSUS,
        /** Para quando probabilidade se estabiliza */
        PROBABILITY_STABILIZATION
    }
    
    /**
     * Métodos de agregação de classificadores.
     */
    public enum AggregationMethod {
        /** Voto majoritário */
        MAJORITY_VOTE,
        /** Média das probabilidades */
        PROBABILITY_AVERAGE,
        /** Média ponderada por confiança */
        WEIGHTED_CONFIDENCE,
        /** Máxima confiança */
        MAX_CONFIDENCE
    }
    
    private final List<BaseClassifier> classifiers;
    private final StoppingStrategy stoppingStrategy;
    private final AggregationMethod aggregationMethod;
    private final double confidenceThreshold;
    private final double marginThreshold;
    private final int minLength;
    private final int maxLength;
    private final int stepSize;
    private final boolean verbose;
    private final Random random;
    
    private String[] classes;
    private boolean fitted;
    
    /**
     * Resultado da classificação precoce.
     */
    public static class EarlyResult {
        private final String predictedClass;
        private final double confidence;
        private final int stoppingTime;
        private final double earliness;
        private final Map<String, Double> classProbs;
        private final List<ClassificationStep> steps;
        
        public EarlyResult(String predictedClass, double confidence, int stoppingTime, 
                          double earliness, Map<String, Double> classProbs, 
                          List<ClassificationStep> steps) {
            this.predictedClass = predictedClass;
            this.confidence = confidence;
            this.stoppingTime = stoppingTime;
            this.earliness = earliness;
            this.classProbs = new HashMap<>(classProbs);
            this.steps = new ArrayList<>(steps);
        }
        
        public String getPredictedClass() { return predictedClass; }
        public double getConfidence() { return confidence; }
        public int getStoppingTime() { return stoppingTime; }
        public double getEarliness() { return earliness; }
        public Map<String, Double> getClassProbabilities() { return classProbs; }
        public List<ClassificationStep> getSteps() { return steps; }
        
        @Override
        public String toString() {
            return String.format("EarlyResult{class='%s', confidence=%.4f, time=%d, earliness=%.4f}", 
                               predictedClass, confidence, stoppingTime, earliness);
        }
    }
    
    /**
     * Representa um passo da classificação.
     */
    public static class ClassificationStep {
        private final int timePoint;
        private final Map<String, Double> probabilities;
        private final double confidence;
        private final boolean shouldStop;
        
        public ClassificationStep(int timePoint, Map<String, Double> probabilities, 
                                double confidence, boolean shouldStop) {
            this.timePoint = timePoint;
            this.probabilities = new HashMap<>(probabilities);
            this.confidence = confidence;
            this.shouldStop = shouldStop;
        }
        
        public int getTimePoint() { return timePoint; }
        public Map<String, Double> getProbabilities() { return probabilities; }
        public double getConfidence() { return confidence; }
        public boolean shouldStop() { return shouldStop; }
    }
    
    /**
     * Construtor com parâmetros padrão.
     */
    public EarlyClassifier() {
        this(new Builder());
    }
    
    private EarlyClassifier(Builder builder) {
        this.classifiers = new ArrayList<>(builder.classifiers);
        this.stoppingStrategy = builder.stoppingStrategy;
        this.aggregationMethod = builder.aggregationMethod;
        this.confidenceThreshold = builder.confidenceThreshold;
        this.marginThreshold = builder.marginThreshold;
        this.minLength = builder.minLength;
        this.maxLength = builder.maxLength;
        this.stepSize = builder.stepSize;
        this.verbose = builder.verbose;
        this.random = new Random(builder.randomSeed);
        this.fitted = false;
        
        if (classifiers.isEmpty()) {
            // Adicionar classificadores padrão
            this.classifiers.add(new ShapeletClassifier());
            this.classifiers.add(new DTWNearestNeighbor());
            this.classifiers.add(new FeatureBasedClassifier());
        }
        
        if (verbose) {
            logger.info("EarlyClassifier configurado: strategy={}, aggregation={}, threshold={}", 
                       stoppingStrategy, aggregationMethod, confidenceThreshold);
        }
    }
    
    /**
     * Builder pattern para configuração flexível.
     */
    public static class Builder {
        private List<BaseClassifier> classifiers = new ArrayList<>();
        private StoppingStrategy stoppingStrategy = StoppingStrategy.CONFIDENCE_THRESHOLD;
        private AggregationMethod aggregationMethod = AggregationMethod.PROBABILITY_AVERAGE;
        private double confidenceThreshold = 0.8;
        private double marginThreshold = 0.3;
        private int minLength = 10;
        private int maxLength = -1; // Auto-detect
        private int stepSize = 1;
        private boolean verbose = false;
        private long randomSeed = System.currentTimeMillis();
        
        public Builder addClassifier(BaseClassifier classifier) {
            this.classifiers.add(classifier);
            return this;
        }
        
        public Builder stoppingStrategy(StoppingStrategy strategy) {
            this.stoppingStrategy = strategy;
            return this;
        }
        
        public Builder aggregationMethod(AggregationMethod method) {
            this.aggregationMethod = method;
            return this;
        }
        
        public Builder confidenceThreshold(double threshold) {
            if (threshold <= 0.0 || threshold > 1.0) {
                throw new IllegalArgumentException("Confidence threshold deve estar entre 0 e 1");
            }
            this.confidenceThreshold = threshold;
            return this;
        }
        
        public Builder marginThreshold(double threshold) {
            if (threshold < 0.0 || threshold > 1.0) {
                throw new IllegalArgumentException("Margin threshold deve estar entre 0 e 1");
            }
            this.marginThreshold = threshold;
            return this;
        }
        
        public Builder minLength(int length) {
            if (length < 1) {
                throw new IllegalArgumentException("Min length deve ser positivo");
            }
            this.minLength = length;
            return this;
        }
        
        public Builder maxLength(int length) {
            this.maxLength = length;
            return this;
        }
        
        public Builder stepSize(int size) {
            if (size < 1) {
                throw new IllegalArgumentException("Step size deve ser positivo");
            }
            this.stepSize = size;
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
        
        public EarlyClassifier build() {
            return new EarlyClassifier(this);
        }
    }
    
    /**
     * Treina o early classifier com dados rotulados.
     */
    public void fit(double[][][] X, String[] y) {
        if (X == null || y == null) {
            throw new IllegalArgumentException("Dados de entrada não podem ser null");
        }
        if (X.length != y.length) {
            throw new IllegalArgumentException("Número de amostras deve ser igual ao número de labels");
        }
        if (X.length == 0) {
            throw new IllegalArgumentException("Dataset não pode estar vazio");
        }
        
        // Extrair classes únicas
        Set<String> uniqueClasses = new HashSet<>(Arrays.asList(y));
        this.classes = uniqueClasses.toArray(new String[0]);
        Arrays.sort(this.classes);
        
        // Determinar maxLength se não especificado
        int actualMaxLength = maxLength;
        if (actualMaxLength <= 0) {
            actualMaxLength = Arrays.stream(X).mapToInt(x -> x.length).max().orElse(100);
        }
        
        if (verbose) {
            logger.info("Treinando EarlyClassifier: {} amostras, {} classes, max_length={}", 
                       X.length, classes.length, actualMaxLength);
        }
        
        // Treinar cada classificador
        for (int i = 0; i < classifiers.size(); i++) {
            BaseClassifier classifier = classifiers.get(i);
            
            if (verbose) {
                logger.info("Treinando classificador {}: {}", i + 1, classifier.getClass().getSimpleName());
            }
            
            try {
                classifier.fit(X, y);
            } catch (Exception e) {
                logger.warn("Erro ao treinar classificador {}: {}", i + 1, e.getMessage());
                // Remove classificador problemático
                classifiers.remove(i);
                i--;
            }
        }
        
        if (classifiers.isEmpty()) {
            throw new RuntimeException("Nenhum classificador foi treinado com sucesso");
        }
        
        this.fitted = true;
        
        if (verbose) {
            logger.info("Treinamento concluído: {} classificadores ativos", classifiers.size());
        }
    }
    
    /**
     * Classifica uma série temporal de forma precoce.
     */
    public EarlyResult predictEarly(double[][] timeSeries) {
        if (!fitted) {
            throw new RuntimeException("Classificador deve ser treinado antes da predição");
        }
        if (timeSeries == null || timeSeries.length == 0) {
            throw new IllegalArgumentException("Série temporal não pode estar vazia");
        }
        
        int seriesLength = timeSeries.length;
        int actualMaxLength = maxLength > 0 ? Math.min(maxLength, seriesLength) : seriesLength;
        
        List<ClassificationStep> steps = new ArrayList<>();
        Map<String, Double> finalProbs = new HashMap<>();
        
        // Iterar sobre prefixos da série temporal
        for (int t = minLength; t <= actualMaxLength; t += stepSize) {
            // Extrair prefixo até tempo t
            double[][] prefix = Arrays.copyOf(timeSeries, t);
            
            // Obter predições de todos os classificadores
            List<Map<String, Double>> classifierProbs = new ArrayList<>();
            for (BaseClassifier classifier : classifiers) {
                try {
                    Map<String, Double> probs = classifier.predictProba(prefix);
                    classifierProbs.add(probs);
                } catch (Exception e) {
                    if (verbose) {
                        logger.warn("Erro na predição do classificador em t={}: {}", t, e.getMessage());
                    }
                    // Skip this classifier for this time point
                }
            }
            
            if (classifierProbs.isEmpty()) {
                continue;
            }
            
            // Agregar probabilidades
            Map<String, Double> aggregatedProbs = aggregateProbabilities(classifierProbs);
            double confidence = calculateConfidence(aggregatedProbs);
            
            // Verificar critério de stopping
            boolean shouldStop = checkStoppingCriterion(aggregatedProbs, confidence, t, steps);
            
            ClassificationStep step = new ClassificationStep(t, aggregatedProbs, confidence, shouldStop);
            steps.add(step);
            
            if (shouldStop) {
                finalProbs = aggregatedProbs;
                break;
            }
            
            // Se chegou ao final sem parar
            if (t >= actualMaxLength - stepSize) {
                finalProbs = aggregatedProbs;
                break;
            }
        }
        
        // Determinar classe predita
        String predictedClass = finalProbs.entrySet().stream()
            .max(Map.Entry.comparingByValue())
            .map(Map.Entry::getKey)
            .orElse(classes[0]);
        
        double confidence = finalProbs.getOrDefault(predictedClass, 0.0);
        int stoppingTime = steps.isEmpty() ? minLength : steps.get(steps.size() - 1).getTimePoint();
        double earliness = 1.0 - (double) stoppingTime / seriesLength;
        
        EarlyResult result = new EarlyResult(predictedClass, confidence, stoppingTime, 
                                           earliness, finalProbs, steps);
        
        if (verbose) {
            logger.info("Early classification: {}", result);
        }
        
        return result;
    }
    
    /**
     * Agrega probabilidades de múltiplos classificadores.
     */
    private Map<String, Double> aggregateProbabilities(List<Map<String, Double>> classifierProbs) {
        Map<String, Double> aggregated = new HashMap<>();
        
        for (String cls : classes) {
            aggregated.put(cls, 0.0);
        }
        
        switch (aggregationMethod) {
            case MAJORITY_VOTE:
                return aggregateByMajorityVote(classifierProbs);
            case PROBABILITY_AVERAGE:
                return aggregateByAverage(classifierProbs);
            case WEIGHTED_CONFIDENCE:
                return aggregateByWeightedConfidence(classifierProbs);
            case MAX_CONFIDENCE:
                return aggregateByMaxConfidence(classifierProbs);
            default:
                return aggregateByAverage(classifierProbs);
        }
    }
    
    private Map<String, Double> aggregateByAverage(List<Map<String, Double>> classifierProbs) {
        Map<String, Double> result = new HashMap<>();
        
        for (String cls : classes) {
            double sum = 0.0;
            int count = 0;
            
            for (Map<String, Double> probs : classifierProbs) {
                if (probs.containsKey(cls)) {
                    sum += probs.get(cls);
                    count++;
                }
            }
            
            result.put(cls, count > 0 ? sum / count : 0.0);
        }
        
        return result;
    }
    
    private Map<String, Double> aggregateByMajorityVote(List<Map<String, Double>> classifierProbs) {
        Map<String, Integer> votes = new HashMap<>();
        
        for (String cls : classes) {
            votes.put(cls, 0);
        }
        
        for (Map<String, Double> probs : classifierProbs) {
            String winner = probs.entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .map(Map.Entry::getKey)
                .orElse(null);
            
            if (winner != null) {
                votes.put(winner, votes.getOrDefault(winner, 0) + 1);
            }
        }
        
        String majorityClass = votes.entrySet().stream()
            .max(Map.Entry.comparingByValue())
            .map(Map.Entry::getKey)
            .orElse(classes[0]);
        
        Map<String, Double> result = new HashMap<>();
        double totalVotes = votes.values().stream().mapToInt(Integer::intValue).sum();
        
        for (String cls : classes) {
            result.put(cls, totalVotes > 0 ? votes.get(cls) / totalVotes : 0.0);
        }
        
        return result;
    }
    
    private Map<String, Double> aggregateByWeightedConfidence(List<Map<String, Double>> classifierProbs) {
        Map<String, Double> weightedSum = new HashMap<>();
        double totalWeight = 0.0;
        
        for (String cls : classes) {
            weightedSum.put(cls, 0.0);
        }
        
        for (Map<String, Double> probs : classifierProbs) {
            double confidence = calculateConfidence(probs);
            totalWeight += confidence;
            
            for (String cls : classes) {
                double prob = probs.getOrDefault(cls, 0.0);
                weightedSum.put(cls, weightedSum.get(cls) + prob * confidence);
            }
        }
        
        Map<String, Double> result = new HashMap<>();
        for (String cls : classes) {
            result.put(cls, totalWeight > 0 ? weightedSum.get(cls) / totalWeight : 0.0);
        }
        
        return result;
    }
    
    private Map<String, Double> aggregateByMaxConfidence(List<Map<String, Double>> classifierProbs) {
        Map<String, Double> bestProbs = null;
        double maxConfidence = 0.0;
        
        for (Map<String, Double> probs : classifierProbs) {
            double confidence = calculateConfidence(probs);
            if (confidence > maxConfidence) {
                maxConfidence = confidence;
                bestProbs = probs;
            }
        }
        
        return bestProbs != null ? bestProbs : new HashMap<>();
    }
    
    /**
     * Calcula confiança baseada na distribuição de probabilidades.
     */
    private double calculateConfidence(Map<String, Double> probabilities) {
        if (probabilities.isEmpty()) return 0.0;
        
        List<Double> probs = new ArrayList<>(probabilities.values());
        probs.sort(Collections.reverseOrder());
        
        if (probs.size() == 1) {
            return probs.get(0);
        }
        
        // Confiança como diferença entre maior e segunda maior probabilidade
        return probs.get(0) - probs.get(1);
    }
    
    /**
     * Verifica se critério de stopping foi atingido.
     */
    private boolean checkStoppingCriterion(Map<String, Double> probabilities, double confidence, 
                                         int timePoint, List<ClassificationStep> steps) {
        switch (stoppingStrategy) {
            case CONFIDENCE_THRESHOLD:
                return confidence >= confidenceThreshold;
            
            case MARGIN_BASED:
                List<Double> probs = new ArrayList<>(probabilities.values());
                probs.sort(Collections.reverseOrder());
                if (probs.size() >= 2) {
                    double margin = probs.get(0) - probs.get(1);
                    return margin >= marginThreshold;
                }
                return false;
            
            case ENSEMBLE_CONSENSUS:
                // Para quando todos os classificadores concordam
                return confidence >= 0.9; // High consensus threshold
            
            case PROBABILITY_STABILIZATION:
                if (steps.size() < 3) return false;
                
                // Verifica se probabilidades se estabilizaram
                ClassificationStep prev = steps.get(steps.size() - 1);
                String topClass = probabilities.entrySet().stream()
                    .max(Map.Entry.comparingByValue())
                    .map(Map.Entry::getKey)
                    .orElse("");
                
                double currentProb = probabilities.getOrDefault(topClass, 0.0);
                double prevProb = prev.getProbabilities().getOrDefault(topClass, 0.0);
                
                return Math.abs(currentProb - prevProb) < 0.05; // Stabilization threshold
            
            default:
                return false;
        }
    }
    
    /**
     * Prediz múltiplas séries temporais.
     */
    public List<EarlyResult> predictEarly(double[][][] X) {
        return Arrays.stream(X)
            .parallel()
            .map(this::predictEarly)
            .collect(ArrayList::new, ArrayList::add, ArrayList::addAll);
    }
    
    /**
     * Avalia o trade-off entre precisão e earliness.
     */
    public EvaluationResult evaluate(double[][][] X_test, String[] y_test) {
        if (!fitted) {
            throw new RuntimeException("Classificador deve ser treinado antes da avaliação");
        }
        
        List<EarlyResult> results = predictEarly(X_test);
        
        int correct = 0;
        double totalEarliness = 0.0;
        double totalConfidence = 0.0;
        
        for (int i = 0; i < results.size(); i++) {
            EarlyResult result = results.get(i);
            String predicted = result.getPredictedClass();
            String actual = y_test[i];
            
            if (predicted.equals(actual)) {
                correct++;
            }
            
            totalEarliness += result.getEarliness();
            totalConfidence += result.getConfidence();
        }
        
        double accuracy = (double) correct / results.size();
        double avgEarliness = totalEarliness / results.size();
        double avgConfidence = totalConfidence / results.size();
        
        return new EvaluationResult(accuracy, avgEarliness, avgConfidence, results);
    }
    
    /**
     * Resultado da avaliação.
     */
    public static class EvaluationResult {
        private final double accuracy;
        private final double averageEarliness;
        private final double averageConfidence;
        private final List<EarlyResult> results;
        
        public EvaluationResult(double accuracy, double averageEarliness, 
                              double averageConfidence, List<EarlyResult> results) {
            this.accuracy = accuracy;
            this.averageEarliness = averageEarliness;
            this.averageConfidence = averageConfidence;
            this.results = new ArrayList<>(results);
        }
        
        public double getAccuracy() { return accuracy; }
        public double getAverageEarliness() { return averageEarliness; }
        public double getAverageConfidence() { return averageConfidence; }
        public List<EarlyResult> getResults() { return results; }
        
        public double getHarmonicMean() {
            if (accuracy == 0 || averageEarliness == 0) return 0.0;
            return 2.0 * accuracy * averageEarliness / (accuracy + averageEarliness);
        }
        
        @Override
        public String toString() {
            return String.format("EvaluationResult{accuracy=%.4f, earliness=%.4f, confidence=%.4f, harmonic=%.4f}", 
                               accuracy, averageEarliness, averageConfidence, getHarmonicMean());
        }
    }
    
    // Getters
    public String[] getClasses() { return classes != null ? classes.clone() : null; }
    public boolean isFitted() { return fitted; }
    public int getNumClassifiers() { return classifiers.size(); }
    public StoppingStrategy getStoppingStrategy() { return stoppingStrategy; }
    public AggregationMethod getAggregationMethod() { return aggregationMethod; }
    public double getConfidenceThreshold() { return confidenceThreshold; }
}
