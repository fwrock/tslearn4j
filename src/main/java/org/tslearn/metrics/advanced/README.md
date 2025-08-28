# Métricas Avançadas para Séries Temporais

Esta pasta contém implementações de métricas avançadas de distância para séries temporais, equivalentes às implementações do Python tslearn.

## Métricas Implementadas

### 1. LCSS (Longest Common Subsequence)

A métrica LCSS mede a similaridade entre séries temporais baseada na maior subsequência comum de pontos que estão dentro de uma tolerância específica.

#### Características:
- **Robusta a ruído**: Tolerante a outliers e ruído
- **Flexível**: Permite diferentes tolerâncias espaciais (epsilon) e temporais (delta)
- **Normalizada**: Resultado entre 0 (idênticas) e 1 (completamente diferentes)

#### Parâmetros:
- `epsilon`: Tolerância espacial para considerar pontos similares
- `delta`: Tolerância temporal máxima para alinhamento
- `verbose`: Ativa logs detalhados

#### Exemplo de uso:
```java
// Configuração básica
LCSS lcss = new LCSS(0.1, 1, false);
double distance = lcss.distance(series1, series2);

// Usando Builder pattern
LCSS lcss = new LCSS.Builder()
    .epsilon(0.2)
    .delta(2)
    .verbose(true)
    .build();

// Auto-configuração
LCSS autoLCSS = new LCSS.Builder()
    .autoEpsilon(series1, series2)
    .autoDelta(series1, series2)
    .build();

// Resultado detalhado
LCSS.LCSSResult result = lcss.distanceWithDetails(series1, series2);
System.out.println("Distância: " + result.getDistance());
System.out.println("Comprimento LCS: " + result.getLcsLength());
System.out.println("Similaridade: " + result.getSimilarity());
```

### 2. MSM (Move-Split-Merge)

A métrica MSM é especializada para comparar séries temporais com diferentes resoluções, usando operações de movimento, divisão e junção.

#### Características:
- **Multi-resolução**: Ideal para séries com diferentes taxas de amostragem
- **Operações específicas**: Move, Split e Merge com custos configuráveis
- **Flexível**: Permite ajustar custos relativos das operações

#### Parâmetros:
- `moveCost`: Custo da operação de movimento
- `splitMergeCost`: Custo das operações de divisão/junção
- `verbose`: Ativa logs detalhados

#### Exemplo de uso:
```java
// Configuração básica
MSM msm = new MSM(1.0, 1.0, false);
double distance = msm.distance(series1, series2);

// Usando Builder pattern
MSM msm = new MSM.Builder()
    .moveCost(0.5)
    .splitMergeCost(1.5)
    .verbose(true)
    .build();

// Auto-configuração
MSM autoMSM = MSM.createAutoConfigured(series1, series2);

// Resultado detalhado
MSM.MSMResult result = msm.distanceWithDetails(series1, series2);
System.out.println("Distância: " + result.getDistance());
System.out.println("Movimentos: " + result.getEstimatedMoves());
System.out.println("Splits/Merges: " + result.getEstimatedSplitsMerges());
System.out.println("Razão de movimentos: " + result.getMoveRatio());
```

### 3. TWE (Time Warp Edit)

A métrica TWE combina Dynamic Time Warping (DTW) com operações de edição, oferecendo controle sobre a rigidez do alinhamento temporal.

#### Características:
- **Híbrida**: Combina DTW com distância de edição
- **Controle de rigidez**: Parâmetro lambda controla flexibilidade temporal
- **Balanceamento**: Parâmetro nu balanceia edição vs. warping

#### Parâmetros:
- `nu`: Peso da penalidade de edição (valores pequenos favorecem edição)
- `lambda`: Controle de rigidez (valores altos = mais rígido)
- `verbose`: Ativa logs detalhados

#### Exemplo de uso:
```java
// Configuração básica
TWE twe = new TWE(0.001, 1.0, false);
double distance = twe.distance(series1, series2);

// Usando Builder pattern
TWE twe = new TWE.Builder()
    .nu(0.005)
    .lambda(2.0)  // Mais rígido
    .verbose(true)
    .build();

// Auto-configuração
TWE autoTWE = TWE.createAutoConfigured(series1, series2);

// Resultado detalhado
TWE.TWEResult result = twe.distanceWithDetails(series1, series2);
System.out.println("Distância: " + result.getDistance());
System.out.println("Componente de edição: " + result.getEditComponent());
System.out.println("Componente de warping: " + result.getWarpComponent());
System.out.println("Razão de edição: " + result.getEditRatio());
```

## Suporte Multivariado

Todas as métricas suportam séries temporais multivariadas:

```java
double[][] multiSeries1 = {{1.0, 0.5}, {2.0, 1.0}, {3.0, 1.5}};
double[][] multiSeries2 = {{1.1, 0.6}, {2.1, 1.1}, {2.9, 1.4}};

LCSS lcss = new LCSS();
double distance = lcss.distance(multiSeries1, multiSeries2);
```

## Auto-configuração

Todas as métricas oferecem métodos de auto-configuração de parâmetros baseados nos dados:

```java
// LCSS
double autoEpsilon = LCSS.calculateAutoEpsilon(series1, series2);
int autoDelta = LCSS.calculateAutoDelta(series1, series2);

// MSM
MSM autoMSM = MSM.createAutoConfigured(series1, series2);

// TWE
TWE autoTWE = TWE.createAutoConfigured(series1, series2);
```

## Casos de Uso Recomendados

### LCSS
- Séries com ruído ou outliers
- Quando robustez é mais importante que precisão exata
- Séries com diferentes comprimentos
- Análise de padrões aproximados

### MSM
- Séries com diferentes resoluções/taxas de amostragem
- Comparação de séries em diferentes escalas temporais
- Quando operações específicas (move/split/merge) têm significado semântico
- Séries hierárquicas ou multi-escala

### TWE
- Quando precisão temporal é importante
- Controle fino sobre flexibilidade vs. rigidez
- Séries que precisam de alinhamento complexo
- Balanceamento entre edição e warping

## Performance

As implementações são otimizadas para performance:

- **LCSS**: O(mn) onde m,n são os comprimentos das séries
- **MSM**: O(mn) com otimizações para cálculo de custos
- **TWE**: O(mn) com cache para componentes de distância

Para séries longas (>500 pontos), considere usar amostragem ou segmentação.

## Testes

Execute os testes unitários:

```bash
./gradlew test --tests "SimpleAdvancedMetricsTest"
```

Execute o exemplo completo:

```bash
./gradlew run
```

## Referências

- Vlachos, M., et al. "Discovering similar multidimensional trajectories." In ICDE, 2002.
- Stefan, A., et al. "The move-split-merge metric for time series." IEEE TKDE, 2013.
- Marteau, P.F. "Time warp edit distance with stiffness adjustment for time series matching." IEEE TPAMI, 2009.
- Tavenard, R., et al. "Tslearn, A Machine Learning Toolkit for Time Series Data." JMLR, 2020.
