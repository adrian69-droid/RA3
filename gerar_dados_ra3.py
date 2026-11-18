"""
Script de geração da base de dados estendida para o RA3.
Mantém estrita aderência ao schema e às categorias observadas no RA1 e RA2:
(event_id, user_id, event_type, value, timestamp)
e adiciona variáveis comportamentais de sessão necessárias para a modelagem supervisionada
(session_duration_sec, pages_viewed, click_count, device_type, discount_applied, is_purchase).
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def gerar_dados_ra3(n_samples=1500, seed=42):
    np.random.seed(seed)
    
    # 1. Definir proporções realistas de tipos de evento
    # Conforme observado no RA1: click, purchase, view
    tipos = ['view', 'click', 'purchase']
    probabilidades = [0.50, 0.30, 0.20]
    event_types = np.random.choice(tipos, size=n_samples, p=probabilidades)
    
    # 2. Usuários (simulando 350 usuários únicos)
    user_ids = np.random.randint(100, 450, size=n_samples)
    
    # 3. Dispositivos
    dispositivos = ['Mobile', 'Desktop', 'Tablet']
    prob_disp = [0.55, 0.35, 0.10]
    device_types = np.random.choice(dispositivos, size=n_samples, p=prob_disp)
    
    # 4. Geração de variáveis comportamentais condicionadas ao tipo de evento
    session_durations = []
    pages_viewed_list = []
    click_counts = []
    values = []
    discounts = []
    
    for et, dev in zip(event_types, device_types):
        # Desconto oferecido aleatoriamente (mais chance no desktop)
        has_discount = 1 if np.random.rand() < (0.35 if dev == 'Desktop' else 0.25) else 0
        discounts.append(has_discount)
        
        if et == 'view':
            # Visualização rápida, sem intenção explícita de compra
            duration = max(10, int(np.random.exponential(scale=65)))
            pages = max(1, int(np.random.poisson(lam=2.5)))
            clicks = max(0, int(np.random.poisson(lam=1.5)))
            val = 0.0
        elif et == 'click':
            # Usuário interagindo com catálogo/links de produtos
            duration = max(30, int(np.random.exponential(scale=140)))
            pages = max(2, int(np.random.poisson(lam=5.0)))
            clicks = max(1, int(np.random.poisson(lam=4.5)))
            # Em alguns cliques pode haver valor estimado/interesse ou 0
            val = round(float(np.random.choice([0.0, 50.0, 100.0, 150.50], p=[0.7, 0.1, 0.1, 0.1])), 2)
        else: # purchase
            # Sessão com funil completo de compra concluído
            duration = max(90, int(np.random.normal(loc=320, scale=95)))
            pages = max(4, int(np.random.poisson(lam=8.5)))
            clicks = max(3, int(np.random.poisson(lam=7.8)))
            # Valores de compras baseados na distribuição observada no RA1 (ex: 890.00, 1250.75)
            # Distribuição log-normal ou gama com média ~750 e variação ampla
            base_val = np.random.gamma(shape=3.5, scale=220)
            if has_discount:
                base_val *= 0.90 # 10% de desconto
            val = round(max(80.0, float(base_val)), 2)
            
        session_durations.append(duration)
        pages_viewed_list.append(pages)
        click_counts.append(clicks)
        values.append(val)
        
    # 5. Timestamps no intervalo de 16/09/2026 a 30/09/2026
    start_date = datetime(2026, 9, 16, 8, 0, 0)
    timestamps = [
        (start_date + timedelta(seconds=int(np.random.uniform(0, 14 * 86400)))).strftime("%Y-%m-%dT%H:%M:%SZ")
        for _ in range(n_samples)
    ]
    
    # 6. Criação do DataFrame
    df = pd.DataFrame({
        'event_id': [f"EVT-{1000 + i}" for i in range(n_samples)],
        'user_id': user_ids,
        'event_type': event_types,
        'value': values,
        'timestamp': timestamps,
        'session_duration_sec': session_durations,
        'pages_viewed': pages_viewed_list,
        'click_count': click_counts,
        'device_type': device_types,
        'discount_applied': discounts,
        'is_purchase': [1 if et == 'purchase' else 0 for et in event_types]
    })
    
    # Ordenar por timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

if __name__ == "__main__":
    df = gerar_dados_ra3(n_samples=1500, seed=42)
    output_path = "data/dados_ra3_analytics.csv"
    df.to_csv(output_path, index=False)
    print(f"Dataset RA3 gerado com sucesso em '{output_path}'.")
    print(f"Total de registros: {len(df)}")
    print(f"Distribuição de event_type:\n{df['event_type'].value_counts(normalize=True)}")
    print(f"Primeiros registros:\n{df.head(3)}")
