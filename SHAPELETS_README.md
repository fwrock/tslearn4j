# Shapelets Implementation - TSLearn4J

## Visão Geral

Esta implementação fornece funcionalidade completa de **Shapelets** para análise de séries temporais em Java, equivalente à biblioteca Python `tslearn`. Shapelets são subsequências discriminativas que podem distinguir efetivamente entre diferentes classes de séries temporais.

## Características Principais

### 🎯 Shapelet Individual
- **Representação de padrões discriminativos**: Cada shapelet representa uma subsequência que caracteriza uma classe
- **Cálculo de distâncias**: Múltiplas métricas de distância para matching
- **Normalização automática**: Z-score normalization para comparações robustas
- **Transformação de datasets**: Conversão de séries temporais em features baseadas em distâncias

### 🔍 ShapeletTransform
- **Descoberta automática**: Algoritmos para encontrar os melhores shapelets discriminativos
- **Múltiplos métodos de seleção**: Information Gain, F-Statistic, Mood's Median, Kruskal-Wallis
- **Estratégias de inicialização**: Random, K-means, Class-balanced
- **Otimizações**: Remoção de shapelets similares, controle de candidatos

## Uso Básico

### Shapelet Individual

```java
// Criar um shapelet
double[][] pattern = {{1.0}, {2.0}, {3.0}, {2.0}, {1.0}};
Shapelet shapelet = new Shapelet(pattern, -1, -1, 0.85, "PatternA");

// Encontrar melhor match em uma série temporal
double[][] timeSeries = /* sua série temporal */;
Shapelet.ShapeletMatch match = shapelet.findBestMatch(timeSeries);
System.out.println("Distância: " + match.getDistance());
System.out.println("Posição: " + match.getPosition());

// Transformar dataset
double[][][] dataset = /* seu dataset */;
double[] distances = shapelet.transform(dataset);
```

### ShapeletTransform Completo

```java
// Configurar transformador
ShapeletTransform transform = new ShapeletTransform.Builder()
    .numShapelets(50)                    // Número de shapelets a descobrir
    .minShapeletLength(3)                // Comprimento mínimo
    .maxShapeletLength(20)               // Comprimento máximo
    .maxCandidates(10000)                // Máximo de candidatos a avaliar
    .selectionMethod(ShapeletTransform.ShapeletSelectionMethod.INFORMATION_GAIN)
    .initializationMethod(ShapeletTransform.InitializationMethod.RANDOM)
    .removeSimilar(true)                 // Remover shapelets similares
    .similarityThreshold(0.1)            // Threshold de similaridade
    .verbose(true)                       // Logs detalhados
    .randomSeed(42L)                     // Seed para reprodutibilidade
    .build();

// Treinar com dados rotulados
double[][][] X_train = /* dados de treino [n_samples][time_length][n_features] */;
String[] y_train = /* labels [n_samples] */;

transform.fit(X_train, y_train);

// Transformar para espaço de features
double[][] X_transformed = transform.transform(X_train);
// X_transformed terá dimensões [n_samples][n_shapelets]

// Analisar shapelets descobertos
List<Shapelet> shapelets = transform.getShapelets();
for (int i = 0; i < Math.min(5, shapelets.size()); i++) {
    Shapelet s = shapelets.get(i);
    System.out.printf("Shapelet %d: Qualidade=%.4f, Comprimento=%d\n", 
                     i+1, s.getQualityScore(), s.getLength());
}
```

## Métodos de Seleção

### Information Gain
Mede a redução na entropia ao usar o shapelet para dividir as classes:
```java
.selectionMethod(ShapeletTransform.ShapeletSelectionMethod.INFORMATION_GAIN)
```

### F-Statistic
Usa análise de variância entre classes:
```java
.selectionMethod(ShapeletTransform.ShapeletSelectionMethod.F_STATISTIC)
```

### Mood's Median Test
Compara medianas entre classes:
```java
.selectionMethod(ShapeletTransform.ShapeletSelectionMethod.MOODS_MEDIAN)
```

### Kruskal-Wallis Test
Teste não-paramétrico para múltiplas classes:
```java
.selectionMethod(ShapeletTransform.ShapeletSelectionMethod.KRUSKAL_WALLIS)
```

## Estratégias de Inicialização

### Random
Extrai candidatos aleatoriamente do dataset:
```java
.initializationMethod(ShapeletTransform.InitializationMethod.RANDOM)
```

### K-means
Usa clustering para encontrar centroides representativos:
```java
.initializationMethod(ShapeletTransform.InitializationMethod.KMEANS)
```

### Class-Balanced
Garante representação equilibrada de todas as classes:
```java
.initializationMethod(ShapeletTransform.InitializationMethod.CLASS_BALANCED)
```

## Exemplo Completo

```java
import org.tslearn.shapelets.*;
import java.util.Random;

public class ShapeletExample {
    public static void main(String[] args) {
        // Gerar dataset sintético
        int nSamples = 100;
        int timeLength = 50;
        int nFeatures = 1;
        
        double[][][] X = new double[nSamples][timeLength][nFeatures];
        String[] y = new String[nSamples];
        Random random = new Random(42);
        
        // Padrões discriminativos
        double[][] patternA = {{1.0}, {2.0}, {1.0}};
        double[][] patternB = {{-1.0}, {-2.0}, {-1.0}};
        
        for (int i = 0; i < nSamples; i++) {
            y[i] = (i % 2 == 0) ? "ClassA" : "ClassB";
            
            // Ruído de fundo
            for (int t = 0; t < timeLength; t++) {
                X[i][t][0] = random.nextGaussian() * 0.3;
            }
            
            // Inserir padrão discriminativo
            double[][] pattern = y[i].equals("ClassA") ? patternA : patternB;
            int insertPos = random.nextInt(timeLength - pattern.length + 1);
            
            for (int j = 0; j < pattern.length; j++) {
                X[i][insertPos + j][0] = pattern[j][0] + random.nextGaussian() * 0.1;
            }
        }
        
        // Configurar e treinar transformador
        ShapeletTransform transform = new ShapeletTransform.Builder()
            .numShapelets(10)
            .minShapeletLength(3)
            .maxShapeletLength(5)
            .selectionMethod(ShapeletTransform.ShapeletSelectionMethod.INFORMATION_GAIN)
            .verbose(true)
            .randomSeed(42L)
            .build();
        
        // Treinar e transformar
        double[][] features = transform.fitTransform(X, y);
        
        System.out.printf("Dataset original: [%d, %d, %d]\n", 
                         X.length, X[0].length, X[0][0].length);
        System.out.printf("Features extraídas: [%d, %d]\n", 
                         features.length, features[0].length);
        System.out.printf("Shapelets descobertos: %d\n", 
                         transform.getNumShapelets());
        
        // Analisar qualidade dos shapelets
        double avgQuality = transform.getShapelets().stream()
            .mapToDouble(Shapelet::getQualityScore)
            .average().orElse(0.0);
        
        System.out.printf("Qualidade média dos shapelets: %.4f\n", avgQuality);
    }
}
```

## Pipeline de Classificação

```java
// 1. Extração de features com Shapelets
ShapeletTransform shapeletTransform = new ShapeletTransform.Builder()
    .numShapelets(100)
    .selectionMethod(ShapeletTransform.ShapeletSelectionMethod.INFORMATION_GAIN)
    .build();

double[][] features_train = shapeletTransform.fitTransform(X_train, y_train);
double[][] features_test = shapeletTransform.transform(X_test);

// 2. Usar features com qualquer classificador
// (Random Forest, SVM, etc. usando as features extraídas)
```

## Parâmetros de Configuração

| Parâmetro | Descrição | Padrão | Recomendações |
|-----------|-----------|---------|---------------|
| `numShapelets` | Número de shapelets a descobrir | 100 | 10-200 dependendo do dataset |
| `minShapeletLength` | Comprimento mínimo dos shapelets | 3 | ≥ 3 para capturar padrões |
| `maxShapeletLength` | Comprimento máximo dos shapelets | Auto | ≤ 50% do comprimento da série |
| `maxCandidates` | Máximo de candidatos a avaliar | 10000 | Aumentar para datasets complexos |
| `selectionMethod` | Método de seleção de qualidade | Information Gain | IG para a maioria dos casos |
| `initializationMethod` | Estratégia de inicialização | Random | Class-balanced para datasets desbalanceados |
| `removeSimilar` | Remover shapelets similares | true | Recomendado para evitar redundância |
| `similarityThreshold` | Threshold de similaridade | 0.1 | 0.05-0.2 dependendo da precisão desejada |

## Performance e Otimizações

### Complexidade Temporal
- **Geração de candidatos**: O(n × m × k) onde n=amostras, m=comprimento, k=candidatos
- **Avaliação de qualidade**: O(k × n × m) para cada candidato
- **Transformação**: O(s × n × m) onde s=número de shapelets selecionados

### Dicas de Performance
1. **Limitar candidatos**: Use `maxCandidates` apropriado para seu dataset
2. **Comprimentos eficientes**: Evite shapelets muito longos
3. **Paralelização**: A avaliação de candidatos pode ser paralelizada
4. **Cache de distâncias**: Reutilize cálculos quando possível

### Uso de Memória
- **Dataset**: O(n × m × f) onde f=features
- **Candidatos**: O(k × l × f) onde l=comprimento médio
- **Features transformadas**: O(n × s)

## Integração com Classificadores

```java
// Extrair features
double[][] train_features = shapeletTransform.fitTransform(X_train, y_train);
double[][] test_features = shapeletTransform.transform(X_test);

// Exemplo com classificador simples (implementar conforme necessário)
// RandomForestClassifier rf = new RandomForestClassifier();
// rf.fit(train_features, y_train);
// String[] predictions = rf.predict(test_features);
```

## Comparação com Python tslearn

| Funcionalidade | TSLearn4J | Python tslearn | Status |
|----------------|-----------|----------------|--------|
| Shapelet individual | ✅ | ✅ | Completo |
| ShapeletTransform | ✅ | ✅ | Completo |
| Information Gain | ✅ | ✅ | Completo |
| Multiple selection methods | ✅ | ✅ | Completo |
| Initialization strategies | ✅ | ✅ | Completo |
| Multivariate support | ✅ | ✅ | Completo |
| Normalization | ✅ | ✅ | Completo |
| Learning Shapelets | 🚧 | ✅ | Em desenvolvimento |

## Recursos Adicionais

- **Testes unitários**: Cobertura completa em `ShapeletTest.java`
- **Exemplos**: `ShapeletTransformExample.java` com casos de uso variados
- **Logging**: Controle detalhado via SLF4J
- **Reprodutibilidade**: Suporte a seeds para resultados determinísticos

## Contribuição

Para contribuir com melhorias:
1. Implemente novos métodos de seleção
2. Otimize algoritmos de busca
3. Adicione paralelização
4. Melhore visualizações
5. Expanda testes de cobertura

---

*Esta implementação mantém compatibilidade conceitual com Python tslearn enquanto aproveita as características da JVM para performance otimizada.*
