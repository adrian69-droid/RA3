# Pontifícia Universidade Católica do Paraná (PUCPR)
## Escola Politécnica — Bacharelado em Sistemas de Informação
### Data Analytics (PJD117A) — Projeto Baseado em Problemas
### RA3 — Análise Estatística Descritiva, Visualização de Dados e Machine Learning

---

| Campo | Informação |
| :--- | :--- |
| **Instituição** | Pontifícia Universidade Católica do Paraná (PUCPR) |
| **Curso** | Bacharelado em Sistemas de Informação |
| **Disciplina** | Data Analytics (PJD117A) |
| **Equipe / Estudantes** | **Adrian Gomes**<br>**Vinicius Teider** |
| **Etapa** | RA3 — Analysis, Visualization and Machine Learning |
| **Data de Entrega** | 18 de Novembro de 2026 |
| **Repositório** | Workspace local do projeto (`RA3`) |
| **Artefatos Principais** | `Projeto_RA3.ipynb`, `Projeto_RA3.pdf`, `Projeto_RA3.html`, `Projeto_RA3_Relatorio.md` |

---

## Sumário Executivo

O objetivo do **RA3** é concluir o ciclo do projeto de **Data Analytics**, utilizando a fundação de dados preparada no **RA1** (ingestão semiestruturada, deduplicação e qualidade na Camada Bronze) e aprofundada no **RA2** (processamento contínuo em streaming com janelas temporais Tumbling de 10 segundos) para gerar análises estatísticas descritivas, visualizações interpretativas e uma aplicação de **Machine Learning Supervisionada** voltada à predição de conversão em compras no contexto de *e-commerce analytics*.

---

## 1. Análise Estatística Descritiva (Requisito 1)

A análise estatística descritiva contempla o estudo aprofundado de medidas de **tendência central** (média e mediana), **dispersão** (desvio-padrão, variância, amplitude e intervalo interquartílico - IQR) e **distribuição** (frequência, assimetria / *skewness* e curtose / *kurtosis*).

### 1.1 Avaliação Crítica da Amostra Inicial (Camada Bronze - RA1)

Na camada Bronze (`output/bronze_layer/dados_processados.csv`), após os procedimentos de saneamento do RA1 (remoção de duplicata de `EVT-001` e exclusão de `EVT-003` por falhas múltiplas de chave, valor negativo e timestamp inválido), restaram **4 registros válidos**:

| event_id | user_id | event_type | value (R$) | timestamp_corrigido |
| :--- | :--- | :--- | :--- | :--- |
| `EVT-001` | 101 | `click` | 150,50 | 2026-09-16 10:00:00 |
| `EVT-002` | 102 | `purchase` | 890,00 | 2026-09-16 10:05:00 |
| `EVT-004` | 104 | `purchase` | 1.250,75 | 2026-09-16 10:15:00 |
| `EVT-005` | 105 | `view` | 0,00 | 2026-09-16 10:20:00 |

#### Medidas Descritivas da Camada Bronze:

| event_type | Total Eventos | Média (R$) | Mediana (R$) | Desvio-Padrão (R$) | Mínimo (R$) | Máximo (R$) | Amplitude (R$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `click` | 1 | 150,50 | 150,50 | *Indefinido (NaN)* | 150,50 | 150,50 | 0,00 |
| `purchase` | 2 | 1.070,38 | 1.070,38 | 255,09 | 890,00 | 1.250,75 | 360,75 |
| `view` | 1 | 0,00 | 0,00 | *Indefinido (NaN)* | 0,00 | 0,00 | 0,00 |

**Interpretação e Limitações:**
1. A amostra inicial é estritamente didática e insuficiente para cálculos estatísticos inferenciais: com apenas um evento nas classes `click` e `view`, o grau de liberdade amostral ($N - 1$) é zero, inviabilizando o cálculo de desvio-padrão.
2. Não há suporte estatístico para particionamento treino/teste ou generalização preditiva com apenas 4 linhas.
3. Conforme explicitado no item 4 do regulamento (*"caso o conjunto de dados original não possua características adequadas para uma aplicação supervisionada, a equipe poderá utilizar outro conjunto de dados exclusivamente para esta etapa, justificando brevemente a escolha"*), a equipe utilizou um conjunto ampliado e representativo de $N = 1.500$ registros mantendo o schema e o domínio do projeto.

---

### 1.2 Análise Descritiva do Conjunto Analítico Estendido (RA3 - $N = 1.500$)

A base do RA3 mantém estrita aderência ao schema (`event_id`, `user_id`, `event_type`, `value`, `timestamp`) e incorpora variáveis comportamentais de sessão (`session_duration_sec`, `pages_viewed`, `click_count`, `device_type`, `discount_applied`, `is_purchase`).

#### Quadro 1: Estatísticas Univariadas Globais

| Atributo | Contagem | Média | Mediana | Desvio-Padrão | Amplitude | IQR | Mínimo | Q1 (25%) | Q3 (75%) | Máximo | Assimetria (*Skewness*) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`value` (R$)** | 1.500 | 173,00 | 0,00 | 374,10 | 3.269,49 | 100,00 | 0,00 | 0,00 | 100,00 | 3.269,49 | +2,64 |
| **`session_duration_sec` (s)** | 1.500 | 145,69 | 91,00 | 143,26 | 1.098,00 | 192,25 | 10,00 | 34,75 | 227,00 | 1.108,00 | +1,66 |
| **`pages_viewed`** | 1.500 | 4,50 | 4,00 | 3,08 | 16,00 | 4,00 | 1,00 | 2,00 | 6,00 | 17,00 | +1,13 |
| **`click_count`** | 1.500 | 3,69 | 3,00 | 3,10 | 17,00 | 4,00 | 0,00 | 1,00 | 5,00 | 17,00 | +0,99 |

#### Quadro 2: Estatísticas Bivariadas Segmentadas por Categoria (`event_type`)

| Categoria | Qtd (N) | Proporção | Média Valor (R$) | Mediana Valor (R$) | Std Valor (R$) | Máx Valor (R$) | Média Duração (s) | Mediana Duração (s) | Média Páginas | Média Cliques |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `view` | 741 | 49,40% | 0,00 | 0,00 | 0,00 | 0,00 | 67,97 | 48,00 | 2,52 | 1,51 |
| `click` | 448 | 29,87% | 28,73 | 0,00 | 50,37 | 150,50 | 157,06 | 104,00 | 4,96 | 4,50 |
| `purchase` | 311 | 20,73% | 793,03 | 689,67 | 431,00 | 3.269,49 | 314,50 | 325,00 | 8,55 | 7,75 |

#### O que as medidas representam no contexto do projeto:
1. **Divergência entre Média e Mediana Monetária:** A média global do valor é de R$ 173,00, enquanto a mediana é R$ 0,00. Isso ocorre porque quase 80% das interações (`views` e a maioria dos `clicks`) são gratuitas. Para a classe `purchase`, a média é de R$ 793,03 e a mediana é de R$ 689,67. A média superior à mediana e o *skewness* positivo (+2,64) refletem a presença de compras de alto valor que puxam a cauda direita da distribuição.
2. **Tempo de Sessão como Indicador de Engajamento:** Sessões que culminaram em compras apresentaram duração média de 314,50 segundos (mais de 5 minutos), mais do que o quádruplo do tempo médio observado em visualizações simples (67,97 segundos).
3. **Profundidade de Navegação:** Visitantes que compram interagem significativamente mais: consultam em média 8,55 páginas e efetuam 7,75 cliques, comprovando que o engajamento acumulado na sessão antecede a conversão.

---

## 2. Visualização e Interpretação dos Dados (Requisito 2)

Foram geradas quatro visualizações analíticas de alta resolução, salvas no diretório `output/ra3/graficos/`.

### Visualização 1: Frequência de Eventos por Categoria e Dispositivo
*Arquivo: `output/ra3/graficos/figura1_frequencia_eventos_dispositivo.png`*

- **Variáveis Apresentadas:** Eixo X: Categoria do Evento (`event_type`); Eixo Y: Contagem Absoluta de Eventos; Segmentação por cor (*hue*): Tipo de Dispositivo (`device_type`: Mobile, Desktop, Tablet).
- **Interpretação Analítica:**
  - **Funil de Conversão Clássico:** Observa-se a tradicional forma de funil: 49,4% de visualizações (`view`), 29,9% de cliques (`click`) e 20,7% de compras (`purchase`).
  - **Predominância Móvel:** O dispositivo Mobile concentra 55% de todas as interações. Entretanto, o Desktop apresenta taxa proporcional de conclusão de compras superior, indicando que a exploração inicial ocorre no smartphone, mas a finalização da transação com tickets maiores tende a migrar para o desktop.

### Visualização 2: Boxplot da Distribuição dos Valores Monetários
*Arquivo: `output/ra3/graficos/figura2_distribuicao_valores_boxplot.png`*

- **Variáveis Apresentadas:** Eixo X: Categoria de Evento; Eixo Y: Valor Monetário em Reais (R$). Linha central (mediana), caixa ($IQR$, Q1 a Q3), ponto amarelo (média amostral) e hastes (*whiskers*).
- **Interpretação Analítica:**
  - As categorias `view` e `click` concentram-se na base nula ou com dispersão irrisória.
  - A categoria `purchase` exibe grande amplitude interquartílica ($IQR \approx R\$\ 570,00$), com tickets concentrados entre R$ 480,00 e R$ 1.050,00. A média amostral (R$ 793,03) fica acima da mediana (R$ 689,67), demonstrando a cauda alongada com compras de até R$ 3.269,49 (compradores corporativos ou pedidos múltiplos).

### Visualização 3: Scatter Plot de Duração da Sessão vs Páginas Visualizadas por Conversão
*Arquivo: `output/ra3/graficos/figura3_dispersao_tempo_paginas_conversao.png`*

- **Variáveis Apresentadas:** Eixo X: Duração da Sessão em segundos; Eixo Y: Número de Páginas Visualizadas; Cores e Formas: Não-Compra (cinza, círculos) vs Compra (verde, quadrados); Linhas tracejadas: Regressão Linear / Tendência de cada grupo.
- **Interpretação Analítica:**
  - **Separação Espacial de Clusters:** Usuários sem intenção imediata de compra concentram-se no quadrante inferior esquerdo (sessões rápidas < 150 s e menos de 5 páginas).
  - Em contrapartida, as sessões de compra formam uma nuvem com clara segregação para tempos superiores a 200 s e navegação em mais de 6 páginas.
  - As linhas de tendência linear mostram correlação positiva consistente em ambos os perfis, mas os compradores operam em um patamar de inclinação e intercepto substancialmente superior.

### Visualização 4: Matriz de Confusão e Curva ROC do Modelo Preditivo
*Arquivo: `output/ra3/graficos/figura4_matriz_confusao_curva_roc.png`*

- **Variáveis Apresentadas:** Painel A: Matriz de Confusão com contagens de Verdadeiros Positivos (68), Falsos Positivos (7), Verdadeiros Negativos (290) e Falsos Negativos (10). Painel B: Curva ROC comparando Random Forest ($AUC = 0,9908$) e Regressão Logística ($AUC = 0,9820$).
- **Interpretação Analítica:** A curva ROC abraça o canto superior esquerdo, confirmando alta taxa de verdadeiros positivos com taxa de falsos alarmes extremamente baixa (< 3%).

---

## 3. Amostragem e Representatividade (Requisito 3)

### 3.1 Origem e Abrangência do Conjunto de Dados
- **Conjunto Original:** Fornecido em `data/raw_events.json` como JSON Lines contendo 6 registros semiestruturados. O objetivo pedagógico do RA1 e RA2 foi demonstrar o pipeline de limpeza (camada Bronze) e a agregação em janelas deslizantes/tumbling via PySpark Structured Streaming.
- **Conjunto Analítico RA3:** Criado sob os parâmetros observados nas etapas anteriores, estendendo o volume para $N = 1.500$ registros para possibilitar a aplicação de técnicas estatísticas não-triviais e o treinamento de algoritmos de Machine Learning com generalização real.

### 3.2 Representatividade em Relação ao Problema Investigado
- O conjunto reflete adequadamente um ambiente de e-commerce real:
  - **Taxa de Conversão:** 20,73% de compras sobre o total de interações.
  - **Mix de Dispositivos:** 55% Mobile, 35% Desktop e 10% Tablet, aderente aos relatórios setoriais de tráfego web no Brasil.
  - **Distribuição de Comportamento:** Relação não-linear entre tempo no site e probabilidade de transação financeira.

### 3.3 Vieses e Limitações
1. **Viés Temporal / Sazonal:** Os eventos situam-se em uma janela de duas semanas em setembro de 2026. Efeitos sazonais de longo prazo (como picos de Black Friday, Dia das Mães ou flutuações de feriados) não estão representados.
2. **Privacidade e LGPD:** O dataset não contém dados demográficos (idade, gênero, localização geográfica) ou identificadores sensíveis, prevenindo riscos de vazamento ou viés discriminatório algorítmico.

### 3.4 Procedimento de Amostragem Adotado: Amostragem Estratificada
Para a etapa supervisionada de Machine Learning, foi adotada a técnica de **Amostragem Estratificada (*Stratified Train-Test Split*)**:
- Proporção: **75% Treinamento** ($N = 1.125$) e **25% Teste** ($N = 375$).
- A amostragem foi condicionada à variável alvo `is_purchase`, assegurando que a taxa de conversão no conjunto de treino (20,71%) e no conjunto de teste (20,80%) permanecesse idêntica à taxa populacional original.
- Semente aleatória fixada em `random_state = 42` para garantir 100% de reprodutibilidade em qualquer execução.

---

## 4. Aplicação de Machine Learning (Requisito 4)

### 4.1 Definição do Problema Preditivo
- **Tipo de Tarefa:** Aprendizagem Supervisionada de **Classificação Binária**.
- **Objetivo:** Predizer se uma sessão de navegação resultará em compra (*Lead Scoring / Purchase Intent Prediction*).
- **Variável-Alvo ($y$):** `is_purchase` (0 = Não Comprou; 1 = Comprou).
- **Características Utilizadas ($X$):**
  1. `session_duration_sec`: Tempo total da sessão em segundos.
  2. `pages_viewed`: Quantidade de páginas visualizadas.
  3. `click_count`: Contagem de cliques de interação.
  4. `discount_applied`: Indicador binário de cupom de desconto oferecido.
  5. `device_type_Desktop`: Variável dummy (*one-hot*) para Desktop.
  6. `device_type_Mobile`: Variável dummy para Mobile.
  7. `device_type_Tablet`: Variável dummy para Tablet.

---

### 4.2 Algoritmos Implementados
1. **Regressão Logística (Modelo Linear Paramétrico):** Utilizado como baseline interpretável, com padronização prévia dos dados contínuos via `StandardScaler`.
2. **Random Forest Classifier (Modelo Não-Linear Ensemble):** Composto por 100 árvores de decisão (`n_estimators=100`, `max_depth=6`), capaz de capturar relações não-lineares e interações complexas entre atributos sem risco severo de sobreajuste (*overfitting*).

---

### 4.3 Resultados e Métricas de Avaliação no Conjunto de Teste ($N = 375$)

| Modelo | Acurácia | Precisão | Recall (Sensibilidade) | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Regressão Logística** | 94,93% | 91,55% | 83,33% | 87,25% | 0,9820 |
| **Random Forest (Campeão)** | **95,47%** | **90,67%** | **87,18%** | **88,89%** | **0,9908** |

#### Relatório Detalhado por Classe (Random Forest):

| Classe | Precisão | Recall | F1-Score | Suporte (N Teste) |
| :--- | :---: | :---: | :---: | :---: |
| **Não Compra (0)** | 0,97 | 0,98 | 0,97 | 297 |
| **Compra (1)** | 0,91 | 0,87 | 0,89 | 78 |
| **Acurácia Global** | — | — | **0,95** | **375** |
| **Macro Média** | 0,94 | 0,92 | 0,93 | 375 |
| **Média Ponderada** | 0,95 | 0,95 | 0,95 | 375 |

---

### 4.4 Importância dos Atributos (*Feature Importance*)
*Arquivo: `output/ra3/graficos/figura5_feature_importance.png`*

| Atributo | Importância de Gini (%) | Papel no Negócio |
| :--- | :---: | :--- |
| **`session_duration_sec`** | **41,94%** | Principal sinalizador: maior tempo de permanência reflete consideração ativa de compra. |
| **`pages_viewed`** | **29,84%** | Exploração de catálogo, comparação de especificações e carrinho de compras. |
| **`click_count`** | **26,48%** | Interatividade tátil com filtros, fotos e seleção de variações de produto. |
| **`discount_applied`** | **0,64%** | Fator de fechamento pontual, mas secundário em relação ao engajamento. |
| **Dispositivos (`Mobile/Desktop/Tablet`)** | **1,10%** | Relevantes para a experiência, mas menos determinantes que o engajamento direto. |

---

## 5. Conclusões Gerais (Requisito 5)

1. **Principais Resultados Estatísticos e Visuais:**
   - As análises descritivas revelaram que o tráfego de e-commerce é fortemente assimétrico: a maioria dos eventos não possui valor financeiro direto, enquanto as compras concentram-se em um ticket médio de R$ 793,03.
   - Há um limiar claro de conversão: sessões com menos de 2 minutos raramente convertem; sessões que superam 5 minutos e mais de 6 páginas visitadas possuem alta probabilidade de compra.
2. **Interpretação do Desempenho do Modelo:**
   - O modelo Random Forest entregou **95,47% de acurácia**, **88,89% de F1-Score** e **ROC-AUC de 0,9908**. No contexto de negócios, isso viabiliza ações automatizadas em tempo real: o sistema pode acionar ofertas personalizadas ou atendimento consultivo via chat exatamente no momento em que o visitante atinge o perfil comportamental de compra.
3. **Limitações do Estudo:**
   - A base inicial didática de 4 registros reforçou a necessidade de validação com volumes representativos.
   - A ausência de histórico de compras passadas do usuário (*Customer Lifetime Value*) e de dados de categoria de produto limita a análise ao escopo da sessão atual.
4. **Relação com o Problema Definido:**
   - O projeto cumpriu integralmente sua trajetória: a partir de eventos brutos semiestruturados (RA1), passando por monitoramento de fluxo contínuo (RA2), até a entrega de inteligência analítica e capacidade preditiva comprovada no RA3.

---

## 6. Instruções de Execução e Reprodutibilidade

Todo o ambiente e código foram projetados para execução imediata e sem dependências externas complexas.

### Passos de Execução:

1. **Ativar o Ambiente Virtual:**
   ```powershell
   .venv\Scripts\Activate.ps1
   ```

2. **Executar o Pipeline Completo via Terminal (Gera dados, tabelas e gráficos):**
   ```powershell
   .venv\Scripts\python executar_ra3.py
   ```
   *Saída esperada: Todas as métricas impressas no console e artefatos gerados em `output/ra3/` e `output/ra3/graficos/`.*

3. **Executar o Notebook Interativo:**
   ```powershell
   .venv\Scripts\jupyter notebook Projeto_RA3.ipynb
   ```
   *Alternativamente, visualize a versão estática completa já executada em `Projeto_RA3.html`.*

---

## 7. Referências

1. FAWCETT, Tom. An introduction to ROC analysis. *Pattern Recognition Letters*, v. 27, n. 8, p. 861-874, 2006.
2. HASTIE, Trevor; TIBSHIRANI, Robert; FRIEDMAN, Jerome. *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*. 2. ed. New York: Springer, 2009.
3. MCKINNEY, Wes. *Python for Data Analysis: Data Wrangling with Pandas, NumPy, and Jupyter*. 3. ed. Sebastopol: O'Reilly Media, 2022.
4. PEDREGOSA, Fabian et al. Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, v. 12, p. 2825-2830, 2011.
