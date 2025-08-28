# Early Classification para Séries Temporais

Esta implementação permite classificar séries temporais antes de observar toda a sequência, ideal para aplicações em tempo real onde decisões rápidas são necessárias.

## Conceito

Early Classification é uma técnica que busca o **equilíbrio entre precisão e earliness** (rapidez de decisão). Ao invés de esperar a série temporal completa, o sistema toma decisões baseadas em prefixos cada vez maiores até atingir um critério de confiança suficiente.

### Vantagens

- ⚡ **Decisões rápidas**: Classificação antes da série completa
- 🎯 **Múltiplas estratégias**: Diferentes critérios de stopping
- 🔄 **Ensemble**: Combina múltiplos classificadores
- 📊 **Trade-off transparente**: Análise clara entre precisão e rapidez
- 🔧 **Configurável**: Ajuste fino de parâmetros

## Componentes Principais

### 1. EarlyClassifier

Classe principal que coordena o processo de early classification:

```java
import org.tslearn.early_classification.*;

// Configuração básica
EarlyClassifier classifier = new EarlyClassifier.Builder()
    .stoppingStrategy(EarlyClassifier.StoppingStrategy.CONFIDENCE_THRESHOLD)
    .confidenceThreshold(0.8)
    .minLength(5)
    .stepSize(2)
    .verbose(true)
    .build();

// Treinar
classifier.fit(trainingData, trainingLabels);

// Classificar com early stopping
EarlyClassifier.EarlyResult result = classifier.predictEarly(timeSeries);
System.out.println("Classe: " + result.getPredictedClass());
System.out.println("Confiança: " + result.getConfidence());
System.out.println("Parou em: " + result.getStoppingTime());
System.out.println("Earliness: " + result.getEarliness());
```

### 2. Estratégias de Stopping

#### CONFIDENCE_THRESHOLD
Para quando a confiança excede um threshold:
```java
EarlyClassifier classifier = new EarlyClassifier.Builder()
    .stoppingStrategy(EarlyClassifier.StoppingStrategy.CONFIDENCE_THRESHOLD)
    .confidenceThreshold(0.8) // Para com 80% de confiança
    .build();
```

#### MARGIN_BASED
Para quando a margem entre classes é suficiente:
```java
EarlyClassifier classifier = new EarlyClassifier.Builder()
    .stoppingStrategy(EarlyClassifier.StoppingStrategy.MARGIN_BASED)
    .marginThreshold(0.3) // Diferença mínima entre 1ª e 2ª classe
    .build();
```

#### PROBABILITY_STABILIZATION
Para quando as probabilidades se estabilizam:
```java
EarlyClassifier classifier = new EarlyClassifier.Builder()
    .stoppingStrategy(EarlyClassifier.StoppingStrategy.PROBABILITY_STABILIZATION)
    .build();
```

#### ENSEMBLE_CONSENSUS
Para quando há consenso entre classificadores:
```java
EarlyClassifier classifier = new EarlyClassifier.Builder()
    .stoppingStrategy(EarlyClassifier.StoppingStrategy.ENSEMBLE_CONSENSUS)
    .build();
```

### 3. Métodos de Agregação

#### PROBABILITY_AVERAGE (padrão)
Média das probabilidades de todos os classificadores:
```java
.aggregationMethod(EarlyClassifier.AggregationMethod.PROBABILITY_AVERAGE)
```

#### MAJORITY_VOTE
Voto majoritário dos classificadores:
```java
.aggregationMethod(EarlyClassifier.AggregationMethod.MAJORITY_VOTE)
```

#### WEIGHTED_CONFIDENCE
Média ponderada pela confiança:
```java
.aggregationMethod(EarlyClassifier.AggregationMethod.WEIGHTED_CONFIDENCE)
```

#### MAX_CONFIDENCE
Usa o classificador com maior confiança:
```java
.aggregationMethod(EarlyClassifier.AggregationMethod.MAX_CONFIDENCE)
```

### 4. Classificadores Base

#### ShapeletClassifier
Baseado em shapelets discriminativos:
```java
EarlyClassifier classifier = new EarlyClassifier.Builder()
    .addClassifier(new ShapeletClassifier())
    .build();
```

#### DTWNearestNeighbor
1-NN com Dynamic Time Warping:
```java
EarlyClassifier classifier = new EarlyClassifier.Builder()
    .addClassifier(new DTWNearestNeighbor())
    .build();
```

#### FeatureBasedClassifier
Baseado em features estatísticas:
```java
EarlyClassifier classifier = new EarlyClassifier.Builder()
    .addClassifier(new FeatureBasedClassifier())
    .build();
```

## Casos de Uso

### 1. Classificação Ultra-Rápida
Para aplicações que precisam de decisões muito rápidas:

```java
EarlyClassifier fastClassifier = new EarlyClassifier.Builder()
    .confidenceThreshold(0.7)  // Threshold mais baixo
    .maxLength(10)             // Máximo 20% da série
    .minLength(3)              // Mínimo absoluto
    .stepSize(1)               // Verifica a cada ponto
    .build();
```

### 2. Classificação Conservadora
Para aplicações que priorizam precisão:

```java
EarlyClassifier conservativeClassifier = new EarlyClassifier.Builder()
    .stoppingStrategy(EarlyClassifier.StoppingStrategy.MARGIN_BASED)
    .marginThreshold(0.4)      // Margem alta
    .minLength(15)             // Observa mais dados
    .stepSize(2)               // Menos verificações
    .build();
```

### 3. Ensemble Customizado
Combinando múltiplos classificadores:

```java
EarlyClassifier ensembleClassifier = new EarlyClassifier.Builder()
    .addClassifier(new DTWNearestNeighbor())
    .addClassifier(new FeatureBasedClassifier())
    .addClassifier(new ShapeletClassifier())
    .stoppingStrategy(EarlyClassifier.StoppingStrategy.ENSEMBLE_CONSENSUS)
    .aggregationMethod(EarlyClassifier.AggregationMethod.WEIGHTED_CONFIDENCE)
    .build();
```

## Avaliação e Métricas

### Avaliação Automática
```java
EarlyClassifier.EvaluationResult evaluation = classifier.evaluate(X_test, y_test);

System.out.println("Precisão: " + evaluation.getAccuracy());
System.out.println("Earliness média: " + evaluation.getAverageEarliness());
System.out.println("Confiança média: " + evaluation.getAverageConfidence());
System.out.println("Média harmônica: " + evaluation.getHarmonicMean());
```

### Análise de Trade-off
```java
double[] thresholds = {0.6, 0.7, 0.8, 0.9, 0.95};

for (double threshold : thresholds) {
    EarlyClassifier classifier = new EarlyClassifier.Builder()
        .confidenceThreshold(threshold)
        .build();
    
    classifier.fit(X_train, y_train);
    EarlyClassifier.EvaluationResult result = classifier.evaluate(X_test, y_test);
    
    System.out.printf("Threshold: %.2f, Accuracy: %.4f, Earliness: %.4f\n",
                     threshold, result.getAccuracy(), result.getAverageEarliness());
}
```

### Análise Detalhada de Predição
```java
EarlyClassifier.EarlyResult result = classifier.predictEarly(timeSeries);

// Evolução passo a passo
for (EarlyClassifier.ClassificationStep step : result.getSteps()) {
    System.out.printf("t=%d: confiança=%.3f, deve_parar=%s\n",
                     step.getTimePoint(), step.getConfidence(), step.shouldStop());
    
    // Probabilidades por classe
    step.getProbabilities().forEach((cls, prob) -> 
        System.out.printf("  %s: %.3f\n", cls, prob));
}
```

## Métricas de Resultado

### EarlyResult
- `getPredictedClass()`: Classe predita
- `getConfidence()`: Confiança da predição
- `getStoppingTime()`: Momento da parada
- `getEarliness()`: Quão cedo parou (0-1)
- `getClassProbabilities()`: Probabilidades finais
- `getSteps()`: Evolução passo a passo

### EvaluationResult
- `getAccuracy()`: Precisão geral
- `getAverageEarliness()`: Earliness média
- `getAverageConfidence()`: Confiança média
- `getHarmonicMean()`: Média harmônica (precisão vs. earliness)

## Performance

### Otimizações Implementadas
- **Processamento paralelo**: Múltiplas predições simultâneas
- **Cache de features**: Reutilização de cálculos
- **Early stopping inteligente**: Evita processamento desnecessário
- **Classificadores otimizados**: DTW com restrições, features eficientes

### Benchmarks Típicos
- **Dataset pequeno (50 séries)**: ~100ms total
- **Dataset médio (500 séries)**: ~1-2s total
- **Dataset grande (5000 séries)**: ~10-20s total

## Casos de Uso Reais

### 1. Monitoramento Industrial
```java
// Detecção precoce de anomalias em sensores
EarlyClassifier anomalyDetector = new EarlyClassifier.Builder()
    .confidenceThreshold(0.85)
    .maxLength(20)  // Máximo 2 segundos de dados
    .minLength(5)   // Mínimo 0.5 segundos
    .build();
```

### 2. Diagnóstico Médico
```java
// Classificação precoce de sinais ECG
EarlyClassifier ecgClassifier = new EarlyClassifier.Builder()
    .stoppingStrategy(EarlyClassifier.StoppingStrategy.MARGIN_BASED)
    .marginThreshold(0.3)
    .addClassifier(new FeatureBasedClassifier())
    .build();
```

### 3. Reconhecimento de Atividades
```java
// Classificação de atividades humanas em tempo real
EarlyClassifier activityClassifier = new EarlyClassifier.Builder()
    .stoppingStrategy(EarlyClassifier.StoppingStrategy.PROBABILITY_STABILIZATION)
    .minLength(10)
    .stepSize(2)
    .build();
```

## Extensibilidade

### Implementando Classificador Customizado
```java
public class CustomClassifier implements BaseClassifier {
    @Override
    public void fit(double[][][] X, String[] y) {
        // Implementar treinamento
    }
    
    @Override
    public String predict(double[][] timeSeries) {
        // Implementar predição
    }
    
    @Override
    public Map<String, Double> predictProba(double[][] timeSeries) {
        // Implementar probabilidades
    }
    
    @Override
    public boolean isFitted() { return fitted; }
    
    @Override
    public String[] getClasses() { return classes; }
}
```

### Usando Classificador Customizado
```java
EarlyClassifier classifier = new EarlyClassifier.Builder()
    .addClassifier(new CustomClassifier())
    .build();
```

## Limitações e Considerações

### Limitações
- **Precisão vs. Rapidez**: Trade-off inerente ao método
- **Dependência de dados**: Qualidade depende dos dados de treino
- **Configuração**: Requer ajuste de parâmetros para domínio específico

### Boas Práticas
- **Validação cruzada**: Para encontrar parâmetros ótimos
- **Análise de trade-off**: Sempre avaliar precisão vs. earliness
- **Ensemble**: Combinar múltiplos classificadores quando possível
- **Monitoramento**: Acompanhar performance em produção

## Referências

- He, Z., et al. "Early classification on time series." Knowledge and Information Systems, 2014.
- Xing, Z., et al. "Early prediction on time series: A nearest neighbor approach." IJCAI, 2009.
- Dachraoui, A., et al. "Early classification of time series as a non myopic sequential decision making problem." ECML PKDD, 2015.
