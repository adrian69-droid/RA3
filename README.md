# PUCPR — Data Analytics (PJD117A)
## RA3 — Análise Estatística Descritiva, Visualização de Dados e Machine Learning

**Pontifícia Universidade Católica do Paraná (PUCPR)**  
**Curso:** Bacharelado em Sistemas de Informação  
**Disciplina:** Data Analytics (PJD117A) — Projeto Baseado em Problemas  
**Equipe:**
- **Adrian Gomes**
- **Vinicius Teider**  
**Etapa:** RA3 — Conclusão do Projeto de Analytics e Aprendizado Supervisionado  
**Data:** Novembro de 2026  

---

## 📌 Apresentação do Projeto

Este repositório contém a entrega completa e independente do **RA3**, focado em:
1. **Análise Estatística Descritiva:** Investigação de medidas de tendência central, dispersão e distribuição tanto na camada Bronze do pipeline quanto na base analítica estendida ($N = 1.500$).
2. **Visualização de Dados:** 5 visualizações gráficas de alta qualidade (300 DPI) com identificação clara de variáveis e interpretações analíticas aprofundadas.
3. **Amostragem e Representatividade:** Discussão crítica da representatividade dos dados, limites da amostra didática inicial, fundamentação da extensão para e-commerce e aplicação de amostragem estratificada.
4. **Machine Learning Supervisionado:** Modelagem de classificação binária da propensão de compra (`is_purchase`) comparando **Random Forest** e **Regressão Logística**, com avaliação abrangente (Acurácia, Precisão, Recall, F1-Score, Curva ROC-AUC e Matriz de Confusão).
5. **Conclusões:** Síntese executiva com impactos de negócio, limitações técnicas e conexões com o problema da disciplina.

---

## 📁 Estrutura de Arquivos

```text
RA3/
├── data/
│   ├── dados_ra3_analytics.csv        # Conjunto de dados analítico (1.500 registros)
│   └── raw_events.json                # Fonte bruta de referência
├── output/
│   ├── bronze_layer/
│   │   └── dados_processados.csv      # Dados limpos da Camada Bronze
│   └── ra3/
│       ├── estatisticas_bronze.csv    # Estatísticas da amostra Bronze
│       ├── estatisticas_geral_ra3.csv # Estatísticas univariadas gerais
│       ├── estatisticas_por_evento.csv# Estatísticas por tipo de evento
│       ├── metricas_modelos.csv       # Tabela comparativa de métricas dos modelos
│       ├── feature_importance.csv     # Grau de importância das features
│       └── graficos/                  # Figuras analíticas em PNG (300 DPI)
│           ├── figura1_frequencia_eventos_dispositivo.png
│           ├── figura2_distribuicao_valores_boxplot.png
│           ├── figura3_dispersao_tempo_paginas_conversao.png
│           ├── figura4_matriz_confusao_curva_roc.png
│           └── figura5_feature_importance.png
├── Projeto_RA3.ipynb                  # Notebook Jupyter oficial executado com todos os outputs
├── Projeto_RA3.pdf                    # Relatório acadêmico completo diagramado em PDF (17 páginas)
├── Projeto_RA3.html                   # Versão HTML do notebook para abertura direta no navegador
├── Projeto_RA3_Relatorio.md           # Relatório formal da entrega em formato Markdown
├── executar_ra3.py                    # Script Python para execução automatizada de ponta a ponta
├── gerar_dados_ra3.py                 # Gerador determinístico do dataset analítico
├── requirements.txt                   # Dependências do projeto
├── .gitignore                         # Arquivos ignorados pelo Git
└── README.md                          # Este documento explicativo
```

---

## 🚀 Como Executar o Projeto

### 1. Clonar o Repositório e Instalar Dependências
```bash
git clone <URL_DO_SEU_REPOSITORIO>
cd RA3
python -m venv .venv
# No Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Instalar pacotes:
pip install -r requirements.txt
```

### 2. Executar via Script Python Automatizado
Para processar os dados, calcular as estatísticas, gerar todos os gráficos e treinar os modelos em um único comando:
```bash
python executar_ra3.py
```
*Todas as tabelas serão salvas em `output/ra3/` e as imagens em `output/ra3/graficos/`.*

### 3. Visualizar o Notebook Interativo
```bash
jupyter notebook Projeto_RA3.ipynb
```
*Ou abra diretamente `Projeto_RA3.html` no seu navegador sem precisar rodar nenhum comando.*

---

## 📊 Principais Resultados

### 1. Estatística Descritiva
- **Valor Transacional (`value`):** Média geral de R$ 173,00 vs. Mediana de R$ 0,00, com forte assimetria positiva (+2,64), visto que ~80% dos eventos não geram faturamento (`views` e a maioria dos `clicks`).
- **Ticket Médio de Compras (`purchase`):** R$ 793,03 (mediana R$ 689,67; desvio-padrão R$ 431,00; máximo R$ 3.269,49).
- **Métricas de Engajamento:** Sessões de compra têm duração média de **314,50 segundos** (> 5 minutos) e navegam por **8,55 páginas**, contra 67,97 segundos e 2,52 páginas para visualizações simples.

### 2. Desempenho de Machine Learning (Classificação de Compra)
- **Target:** `is_purchase` (0 = Não Comprou, 1 = Comprou).
- **Amostragem:** Estratificada em 75% Treino ($N = 1.125$) e 25% Teste ($N = 375$).

| Modelo | Acurácia | Precisão | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Regressão Logística** | 94,93% | 91,55% | 83,33% | 87,25% | 0,9820 |
| **Random Forest (Campeão)** | **95,47%** | **90,67%** | **87,18%** | **88,89%** | **0,9908** |

### 3. Importância das Características (Feature Importance)
- **Duração da Sessão (`session_duration_sec`):** 41,94%
- **Páginas Visualizadas (`pages_viewed`):** 29,84%
- **Cliques (`click_count`):** 26,48%
- **Descontos e Dispositivo:** ~1,74%
