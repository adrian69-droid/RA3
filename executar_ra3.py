"""
Script de Execução Completa do RA3 - Analysis, Visualization and Machine Learning
PUCPR - Bacharelado em Sistemas de Informação - Data Analytics (PJD117A)

Este script executa:
1. Análise Estatística Descritiva (Camada Bronze e Conjunto Analítico RA3)
2. Geração e Salvamento das Visualizações (PNG em alta resolução)
3. Avaliação de Amostragem e Divisão Estratificada
4. Treinamento e Avaliação de Modelos de Machine Learning (Classificação)
5. Exportação dos Resultados e Métricas para Documentação
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

# Configuração de estilo visual dos gráficos
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11

OUTPUT_DIR = "output/ra3"
GRAFICOS_DIR = os.path.join(OUTPUT_DIR, "graficos")
os.makedirs(GRAFICOS_DIR, exist_ok=True)

def carregar_dados():
    # 1. Carregar Bronze Layer
    caminho_bronze = "output/bronze_layer/dados_processados.csv"
    if os.path.exists(caminho_bronze):
        df_bronze = pd.read_csv(caminho_bronze)
    else:
        df_bronze = None
        
    # 2. Carregar Dataset RA3
    caminho_ra3 = "data/dados_ra3_analytics.csv"
    df_ra3 = pd.read_csv(caminho_ra3)
    return df_bronze, df_ra3

def executar_analise_estatistica(df_bronze, df_ra3):
    print("=" * 70)
    print("1. ANÁLISE ESTATÍSTICA DESCRITIVA")
    print("=" * 70)
    
    # Análise Bronze Layer
    print("\n--- AMOSTRA ORIGINAL BRONZE LAYER (4 registros) ---")
    if df_bronze is not None:
        stats_bronze = df_bronze.groupby('event_type')['value'].agg(
            total='count',
            media='mean',
            mediana='median',
            desvio_padrao='std',
            minimo='min',
            maximo='max',
            amplitude=lambda x: x.max() - x.min()
        ).reset_index()
        print(stats_bronze.to_string(index=False))
        stats_bronze.to_csv(os.path.join(OUTPUT_DIR, "estatisticas_bronze.csv"), index=False)
    
    # Análise Dataset RA3
    print("\n--- CONJUNTO ANALÍTICO RA3 (1.500 registros) ---")
    colunas_numericas = ['value', 'session_duration_sec', 'pages_viewed', 'click_count']
    
    stats_geral = df_ra3[colunas_numericas].describe().T
    stats_geral['amplitude'] = stats_geral['max'] - stats_geral['min']
    stats_geral['iqr'] = stats_geral['75%'] - stats_geral['25%']
    stats_geral['mediana'] = df_ra3[colunas_numericas].median()
    stats_geral['variancia'] = df_ra3[colunas_numericas].var()
    stats_geral['skewness'] = df_ra3[colunas_numericas].skew()
    stats_geral['kurtosis'] = df_ra3[colunas_numericas].kurtosis()
    
    stats_geral = stats_geral[['count', 'mean', 'mediana', 'std', 'amplitude', 'iqr', 'min', '25%', '75%', 'max', 'skewness']]
    print("\nEstatísticas Univariadas Globais:")
    print(stats_geral.round(2).to_string())
    stats_geral.round(2).to_csv(os.path.join(OUTPUT_DIR, "estatisticas_geral_ra3.csv"))
    
    # Agrupado por tipo de evento
    print("\nEstatísticas por Categoria de Evento (event_type):")
    stats_tipo = df_ra3.groupby('event_type').agg(
        qtd=('event_id', 'count'),
        proporcao=('event_id', lambda x: len(x) / len(df_ra3)),
        media_valor=('value', 'mean'),
        mediana_valor=('value', 'median'),
        std_valor=('value', 'std'),
        max_valor=('value', 'max'),
        media_duracao=('session_duration_sec', 'mean'),
        mediana_duracao=('session_duration_sec', 'median'),
        media_paginas=('pages_viewed', 'mean'),
        media_cliques=('click_count', 'mean')
    ).reset_index()
    print(stats_tipo.round(2).to_string(index=False))
    stats_tipo.round(2).to_csv(os.path.join(OUTPUT_DIR, "estatisticas_por_evento.csv"), index=False)
    
    return stats_geral, stats_tipo

def gerar_visualizacoes(df_ra3):
    print("\n" + "=" * 70)
    print("2. GERAÇÃO DE VISUALIZAÇÕES E INTERPRETAÇÕES")
    print("=" * 70)
    
    # Paleta de cores corporativa/acadêmica
    cores = ['#2b5c8f', '#4682b4', '#87ceeb', '#e26d5c', '#38b000']
    
    # ----------------------------------------------------
    # Visualização 1: Distribuição de Frequência de Eventos por Dispositivo
    # ----------------------------------------------------
    plt.figure(figsize=(9, 5.5))
    ax = sns.countplot(
        data=df_ra3,
        x='event_type',
        hue='device_type',
        palette=['#1f77b4', '#ff7f0e', '#2ca02c'],
        order=['view', 'click', 'purchase']
    )
    plt.title("Figura 1: Distribuição de Frequência de Eventos por Categoria e Dispositivo", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Categoria de Evento (event_type)", fontsize=11, fontweight='semibold')
    plt.ylabel("Contagem Absoluta de Eventos", fontsize=11, fontweight='semibold')
    plt.legend(title="Dispositivo", title_fontsize='10', fontsize=10)
    
    # Anotar percentuais e valores nas barras
    total_eventos = len(df_ra3)
    for p in ax.patches:
        altura = p.get_height()
        if altura > 0:
            pct = 100 * altura / total_eventos
            ax.annotate(f"{int(altura)}\n({pct:.1f}%)",
                        (p.get_x() + p.get_width() / 2., altura / 2),
                        ha='center', va='center', fontsize=9, color='white', fontweight='bold')
            
    plt.tight_layout()
    fig1_path = os.path.join(GRAFICOS_DIR, "figura1_frequencia_eventos_dispositivo.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"-> Salvo: {fig1_path}")
    
    # ----------------------------------------------------
    # Visualização 2: Boxplot da Distribuição de Valores Monetários por Tipo de Evento
    # ----------------------------------------------------
    plt.figure(figsize=(9, 5.5))
    df_box = df_ra3.copy()
    ax = sns.boxplot(
        data=df_box,
        x='event_type',
        y='value',
        hue='event_type',
        palette=['#a8dadc', '#457b9d', '#e63946'],
        order=['view', 'click', 'purchase'],
        showmeans=True,
        legend=False,
        meanprops={"marker":"o", "markerfacecolor":"yellow", "markeredgecolor":"black", "markersize":"8"}
    )
    plt.title("Figura 2: Distribuição dos Valores Monetários (R$) por Tipo de Evento", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Categoria de Evento", fontsize=11, fontweight='semibold')
    plt.ylabel("Valor Transacional (R$)", fontsize=11, fontweight='semibold')
    
    # Adicionar legenda para a média
    plt.plot([], [], marker='o', color='yellow', markeredgecolor='black', linestyle='None', label='Média')
    plt.legend(loc='upper left', frameon=True)
    
    plt.tight_layout()
    fig2_path = os.path.join(GRAFICOS_DIR, "figura2_distribuicao_valores_boxplot.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"-> Salvo: {fig2_path}")
    
    # ----------------------------------------------------
    # Visualização 3: Scatter Plot - Duração da Sessão vs Páginas com Conversão
    # ----------------------------------------------------
    plt.figure(figsize=(9.5, 5.5))
    ax = sns.scatterplot(
        data=df_ra3,
        x='session_duration_sec',
        y='pages_viewed',
        hue='is_purchase',
        palette={0: '#6c757d', 1: '#2b9348'},
        alpha=0.75,
        s=60,
        style='is_purchase',
        markers={0: 'o', 1: 's'}
    )
    
    # Linha de tendência linear para sessões de compra vs não-compra
    sns.regplot(
        data=df_ra3[df_ra3['is_purchase'] == 1],
        x='session_duration_sec',
        y='pages_viewed',
        scatter=False,
        ax=ax,
        color='#1b4332',
        line_kws={'linestyle':'--', 'linewidth': 2}
    )
    sns.regplot(
        data=df_ra3[df_ra3['is_purchase'] == 0],
        x='session_duration_sec',
        y='pages_viewed',
        scatter=False,
        ax=ax,
        color='#495057',
        line_kws={'linestyle':':', 'linewidth': 2}
    )
    
    plt.title("Figura 3: Relação entre Duração da Sessão, Páginas Visualizadas e Conversão", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Duração da Sessão (segundos)", fontsize=11, fontweight='semibold')
    plt.ylabel("Páginas Visualizadas na Sessão", fontsize=11, fontweight='semibold')
    
    # Customizar legendas
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#6c757d', markersize=8, label='Não Compra (0)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#2b9348', markersize=8, label='Compra (1)'),
        Line2D([0], [0], color='#1b4332', linestyle='--', linewidth=2, label='Tendência (Compra)'),
        Line2D([0], [0], color='#495057', linestyle=':', linewidth=2, label='Tendência (Não Compra)')
    ]
    ax.legend(handles=legend_elements, title="Resultado da Sessão", loc='upper left')
    
    plt.tight_layout()
    fig3_path = os.path.join(GRAFICOS_DIR, "figura3_dispersao_tempo_paginas_conversao.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"-> Salvo: {fig3_path}")
    
    return [fig1_path, fig2_path, fig3_path]

def executar_machine_learning(df_ra3):
    print("\n" + "=" * 70)
    print("3. MODELAGEM DE MACHINE LEARNING SUPERVISIONADA")
    print("=" * 70)
    
    # 1. Definição do Problema e Atributos
    # Problema: Prever se a sessão/evento converterá em compra (is_purchase: 0 ou 1)
    df_ml = df_ra3.copy()
    
    # One-hot encoding de device_type
    df_ml = pd.get_dummies(df_ml, columns=['device_type'], drop_first=False, dtype=int)
    
    feature_cols = [
        'session_duration_sec',
        'pages_viewed',
        'click_count',
        'discount_applied',
        'device_type_Desktop',
        'device_type_Mobile',
        'device_type_Tablet'
    ]
    
    X = df_ml[feature_cols]
    y = df_ml['is_purchase']
    
    print(f"Tamanho total da base para ML: {len(df_ml)} registros")
    print(f"Variável alvo (Target): 'is_purchase'")
    print(f"Distribuição do Target:\n{y.value_counts(normalize=True).round(4).to_dict()}")
    print(f"Características preditoras (Features): {feature_cols}")
    
    # 2. Divisão Estratificada Treino e Teste (75% treino, 25% teste)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"\nPartição dos dados:")
    print(f"- Conjunto de Treinamento: {len(X_train)} amostras (75%)")
    print(f"- Conjunto de Teste: {len(X_test)} amostras (25%)")
    print(f"Taxa de conversão no Treino: {y_train.mean():.4f}")
    print(f"Taxa de conversão no Teste:  {y_test.mean():.4f}")
    
    # 3. Pré-processamento: Padronização para Regressão Logística
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    num_cols = ['session_duration_sec', 'pages_viewed', 'click_count']
    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])
    
    # 4. Treinamento de Modelos
    # Modelo 1: Regressão Logística
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    y_pred_lr = lr_model.predict(X_test_scaled)
    y_prob_lr = lr_model.predict_proba(X_test_scaled)[:, 1]
    
    # Modelo 2: Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
    
    # 5. Avaliação Comparativa de Métricas
    metricas = {
        'Modelo': ['Regressão Logística', 'Random Forest'],
        'Acurácia': [accuracy_score(y_test, y_pred_lr), accuracy_score(y_test, y_pred_rf)],
        'Precisão': [precision_score(y_test, y_pred_lr), precision_score(y_test, y_pred_rf)],
        'Recall (Sensibilidade)': [recall_score(y_test, y_pred_lr), recall_score(y_test, y_pred_rf)],
        'F1-Score': [f1_score(y_test, y_pred_lr), f1_score(y_test, y_pred_rf)],
        'ROC-AUC': [roc_auc_score(y_test, y_prob_lr), roc_auc_score(y_test, y_prob_rf)]
    }
    df_metricas = pd.DataFrame(metricas)
    print("\nQuadro Comparativo de Desempenho dos Modelos:")
    print(df_metricas.round(4).to_string(index=False))
    df_metricas.round(4).to_csv(os.path.join(OUTPUT_DIR, "metricas_modelos.csv"), index=False)
    
    # Relatório detalhado do modelo Random Forest (modelo escolhido)
    print("\nRelatório de Classificação Detalhado (Random Forest):")
    print(classification_report(y_test, y_pred_rf, target_names=['Não Compra', 'Compra']))
    
    # 6. Geração da Figura 4: Matriz de Confusão e Curva ROC
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # Subplot 1: Matriz de Confusão
    cm = confusion_matrix(y_test, y_pred_rf)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0],
                xticklabels=['Previsto Não', 'Previsto Sim'],
                yticklabels=['Real Não', 'Real Sim'],
                annot_kws={'size': 14, 'weight': 'bold'})
    axes[0].set_title("Matriz de Confusão (Random Forest)", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Classe Verdadeira", fontsize=11)
    axes[0].set_xlabel("Classe Prevista", fontsize=11)
    
    # Subplot 2: Curva ROC
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
    axes[1].plot(fpr_rf, tpr_rf, color='#2ca02c', lw=2.5, label=f'Random Forest (AUC = {roc_auc_score(y_test, y_prob_rf):.3f})')
    axes[1].plot(fpr_lr, tpr_lr, color='#1f77b4', lw=2, linestyle='--', label=f'Regressão Logística (AUC = {roc_auc_score(y_test, y_prob_lr):.3f})')
    axes[1].plot([0, 1], [0, 1], color='grey', lw=1.5, linestyle=':')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Taxa de Falsos Positivos (1 - Especificidade)', fontsize=11)
    axes[1].set_ylabel('Taxa de Verdadeiros Positivos (Recall)', fontsize=11)
    axes[1].set_title('Curva ROC Comparativa', fontsize=12, fontweight='bold')
    axes[1].legend(loc="lower right", frameon=True)
    
    plt.suptitle("Figura 4: Desempenho da Avaliação Preditiva de Machine Learning", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig4_path = os.path.join(GRAFICOS_DIR, "figura4_matriz_confusao_curva_roc.png")
    plt.savefig(fig4_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"-> Salvo: {fig4_path}")
    
    # 7. Importância das Features
    importancias = pd.DataFrame({
        'Feature': feature_cols,
        'Importancia': rf_model.feature_importances_
    }).sort_values('Importancia', ascending=False)
    print("\nImportância das Variáveis (Random Forest):")
    print(importancias.round(4).to_string(index=False))
    importancias.round(4).to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)
    
    # Gráfico de Feature Importance
    plt.figure(figsize=(8, 4.5))
    sns.barplot(data=importancias, x='Importancia', y='Feature', hue='Feature', legend=False, palette='viridis')
    plt.title("Figura 5: Importância Relativa dos Atributos Preditivos (Gini Importance)", fontsize=12, fontweight='bold', pad=12)
    plt.xlabel("Grau de Importância", fontsize=10)
    plt.ylabel("Atributo (Feature)", fontsize=10)
    plt.tight_layout()
    fig5_path = os.path.join(GRAFICOS_DIR, "figura5_feature_importance.png")
    plt.savefig(fig5_path, dpi=300)
    plt.close()
    print(f"-> Salvo: {fig5_path}")
    
    return df_metricas, importancias

def main():
    print("Iniciando Pipeline de Análise e Machine Learning do RA3...\n")
    df_bronze, df_ra3 = carregar_dados()
    stats_geral, stats_tipo = executar_analise_estatistica(df_bronze, df_ra3)
    gerar_visualizacoes(df_ra3)
    df_metricas, importancias = executar_machine_learning(df_ra3)
    print("\n" + "=" * 70)
    print("PIPELINE DO RA3 CONCLUÍDO COM SUCESSO!")
    print(f"Artefatos e figuras gravados em: '{OUTPUT_DIR}' e '{GRAFICOS_DIR}'")
    print("=" * 70)

if __name__ == "__main__":
    main()
