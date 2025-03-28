import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import math
import io
import requests
import time
from datetime import timedelta
from statsmodels.sandbox.distributions.extras import pdf_mvsk
import matplotlib.pyplot as plt

# ----------------------- FUNÇÕES AUXILIARES -----------------------
def calculate_rsi_manual(data: pd.DataFrame, period: int = 14) -> pd.Series:
    '''Calcula o Índice de Força Relativa (RSI) manualmente.'''
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def gordon_model(dividend: float, growth_rate: float, discount_rate: float) -> float:
    '''Calcula o preço justo usando o modelo de Gordon.'''
    if growth_rate is not None and growth_rate < discount_rate:
        price_gordon = dividend / (discount_rate - growth_rate)
        return price_gordon.item()
    else:
        return None

def calculate_npv(cash_flows: np.ndarray, discount_rate: float) -> float:
    '''Calcula o Valor Presente Líquido (NPV) dos fluxos de caixa.'''
    if len(cash_flows) == 0 or discount_rate <= -1:
        return 0  # Evita erro de divisão por zero ou entrada inválida
    periods = np.arange(1, len(cash_flows) + 1)  
    npv = np.sum(cash_flows / ((1 + discount_rate) ** periods))
    return npv

def dcf_model(cash_flows: list, discount_rate: float, terminal_growth_rate: float) -> float:
    '''Modelo DCF que inclui um valor terminal.'''
    if not cash_flows or discount_rate <= terminal_growth_rate:
        return None  # Evita erro de divisão por zero no valor terminal

    pv = sum(cf / ((1 + discount_rate) ** (i + 1)) for i, cf in enumerate(cash_flows))

    last_fcf = cash_flows[-1] * (1 + terminal_growth_rate)  
    terminal_value = last_fcf / (discount_rate - terminal_growth_rate)

    terminal_pv = terminal_value / ((1 + discount_rate) ** len(cash_flows))
    
    return pv + terminal_pv

def gera_cenarios(price_df: pd.DataFrame, vol_lb: int = 66, n_iter: int = 1000, dias_sim: int = 30) -> pd.DataFrame:
    '''
    Gera cenários de preços futuros com base nos retornos históricos.
    Retorna um DataFrame onde cada coluna é um cenário simulado.
    '''
    returns = price_df['Close'].pct_change().dropna()[-vol_lb:]
    std, skew, kurtosis = returns.std(), returns.skew(), returns.kurtosis()
    pdf = pdf_mvsk([0, std**2, skew, kurtosis])
    PX_0 = price_df['Close'].iat[-1]
    X = np.linspace(-10 * std, 10 * std, 10000)
    P_X = np.cumsum(pdf(X)) / np.sum(pdf(X))

    rand_matrix = np.random.rand(dias_sim, n_iter)
    retornos = np.empty_like(rand_matrix)
    for i in range(dias_sim):
        retornos[i, :] = np.interp(rand_matrix[i, :], P_X, X)
    sim_prices = PX_0 * np.cumprod(1 + retornos, axis=0)
    
    index = [price_df.index[-1] + timedelta(days=i) for i in range(1, dias_sim + 1)]
    cenarios = pd.DataFrame(sim_prices, index=index)

    return cenarios

def generate_recommendation(fair_price: float, current_price: float, future_prices: pd.Series) -> tuple:
    '''Gera recomendação baseada no preço justo, atual e simulação futura.'''
    if fair_price is None or current_price is None or future_prices is None:
        return 'Dados insuficientes', 'Não é possível gerar uma recomendação.'
    margin = fair_price - current_price
    future_growth = ((future_prices.iloc[-1] - current_price) / current_price) * 100
    if margin > 0 and future_growth > 0:
        return 'Forte Oportunidade', 'Ativo subvalorizado com potencial de crescimento.'
    elif margin > 0:
        return 'Oportunidade Moderada', 'Ativo subvalorizado, porém o crescimento é incerto.'
    elif future_growth > 0:
        return 'Oportunidade Moderada', 'Cautela: ativo pode estar sobrevalorizado, mas com potencial de crescimento.'
    else:
        return 'Fraca Oportunidade', 'Evitar investimento: ativo sobrevalorizado e com baixo potencial.'

def create_pdf_report(filename: str, report_content: dict) -> None:
    '''Gera relatório PDF com dados e gráficos do ativo.'''
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch

    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [
        Paragraph('Relatório de Análise de Ações', styles['Title']),
        Spacer(1, 12),
        Paragraph(f'Análise detalhada do ativo {report_content['ticker']}.', styles['BodyText']),
        Spacer(1, 12)
    ]
    
    for modelo, valor in zip(['Gordon', 'DCF'],
                             [report_content.get('fair_price_gordon'), report_content.get('fair_price_dcf')]):
        txt = f'Preço justo ({modelo}): {valor:.2f}' if valor and not math.isnan(valor) else f'Preço justo ({modelo}): N/A'
        story.append(Paragraph(txt, styles['BodyText']))
    story.append(Spacer(1, 12))
    
    story.append(Paragraph(f'Valor atual: {report_content['current_price']:.2f}', styles['BodyText']))
    if report_content.get('future_prices') is not None:
        profit = ((report_content['future_prices'].iloc[-1] - report_content['current_price']) / report_content['current_price'] * 100)
        story.append(Paragraph(f'Previsão de lucro: {profit:.2f}%', styles['BodyText']))
    else:
        story.append(Paragraph('Previsão de lucro: N/A', styles['BodyText']))
    story.append(Spacer(1, 12))
    
    story.append(Paragraph(f'Força da Oportunidade: {report_content['strength']}', styles['BodyText']))
    story.append(Paragraph(f'Recomendação: {report_content['recommendation']}', styles['BodyText']))
    story.append(Spacer(1, 12))
    
    if report_content.get('future_prices') is not None:
        try:
            plt.figure(figsize=(6, 4))
            plt.plot(report_content['future_prices'], label='Preços Futuros')
            plt.title('Previsão de Preços Futuros')
            plt.xlabel('Dias')
            plt.ylabel('Preço')
            plt.legend()
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            story.append(Image(buf, width=6*inch, height=4*inch))
            plt.close()
        except Exception:
            story.append(Paragraph('Gráfico de preços futuros indisponível.', styles['BodyText']))
    else:
        story.append(Paragraph('Gráfico de preços futuros indisponível.', styles['BodyText']))
    
    story.append(Paragraph('Relatório gerado automaticamente.', styles['BodyText']))
    doc.build(story)

def calcular_indicadores(df):
    '''Calcula indicadores a partir do DataFrame consolidado'''
    
    df_filtrado = df.replace(0, np.nan).dropna(subset=[
        'Current Liabilities', 'Total Debt', 'Stockholders Equity', 'Total Revenue', 'Total Assets'
    ])

    indicadores_lista = []
    for _, row in df_filtrado.iterrows():
        indicadores = {}

        # Liquidez e Solvência
        indicadores['solvencia de caixa'] = row.get('Cash And Cash Equivalents', 0) / row.get('Current Liabilities', 1)
        indicadores['liquidez seca'] = (row.get('Current Assets', 0) - row.get('Inventory', 0)) / row.get('Current Liabilities', 1)
        indicadores['liquidez operacional'] = row.get('Operating Cash Flow', 0) / row.get('Current Liabilities', 1)
        indicadores['liquidez corrente'] = row.get('Current Assets', 0) / row.get('Current Liabilities', 1)
        indicadores['liquidez geral'] = (row.get('Current Assets', 0) + row.get('Current Liabilities', 0)) / (
            row.get('Current Liabilities', 1) + row.get('Total Non Current Liabilities Net Minority Interest', 0))

        # Endividamento
        indicadores['endividamento financeiro'] = row.get('Total Debt', 0) / (
            row.get('Total Debt', 0) + row.get('Stockholders Equity', 1))
        indicadores['participacao no pl'] = row.get('Stockholders Equity', 0) / (
            row.get('Total Debt', 0) + row.get('Stockholders Equity', 1))
        indicadores['endividamento de curto prazo'] = row.get('Current Debt', 0) / row.get('Total Debt', 1)
        indicadores['endividamento de longo prazo'] = row.get('Long Term Debt', 0) / row.get('Total Debt', 1)

        # Cobertura de juros
        indicadores['cobertura de juros'] = row.get('Operating Income', 0) / row.get('Interest Expense', 1)

        # Multiplicadores financeiros
        indicadores['pl'] = row.get('Total Assets', 0) / row.get('Stockholders Equity', 1)
        indicadores['capital de terceiros'] = row.get('Total Assets', 0) / row.get('Total Debt', 1)

        # Margens
        indicadores['margem bruta %'] = (row.get('Gross Profit', 0) / row.get('Total Revenue', 1)) * 100
        indicadores['margem operacional %'] = (row.get('Operating Income', 0) / row.get('Total Revenue', 1)) * 100
        indicadores['margem liquida %'] = (row.get('Net Income', 0) / row.get('Total Revenue', 1)) * 100

        # Rentabilidade
        indicadores['roa %'] = (row.get('Net Income', 0) / row.get('Total Assets', 1)) * 100
        indicadores['roe %'] = (row.get('Net Income', 0) / row.get('Stockholders Equity', 1)) * 100
        
        indicadores_lista.append(indicadores)

    df_indicadores = pd.DataFrame(indicadores_lista, index=df_filtrado.index)
    df_indicadores = df_indicadores.apply(pd.to_numeric, errors='coerce')

    df_indicadores.index = pd.to_datetime(df_indicadores.index).strftime('%d/%m/%Y')

    return df_indicadores


# ----------------------- FUNÇÕES DE CACHE -----------------------
@st.cache_data(ttl=3600)
def get_stock_history(ticker: str, period: str = '5y') -> pd.DataFrame:
    retries = 5
    for _ in range(retries):
        try:
            return yf.Ticker(ticker).history(period=period)
        except Exception as e:
            if 'RateLimit' in str(e):
                st.write('Limite de requisições atingido. Tentando novamente em 30 segundos...')
                time.sleep(30)
            else:
                st.write(f'Erro inesperado ao obter os dados: {e}')
                break
    st.write('Falha ao obter os dados após várias tentativas.')
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_yf(ticker: str) -> pd.DataFrame:
    '''
    Retorna df, df_info, df_dividendos contendo os dados do ticker: 'financials, balance_sheet e cashflow', 'Informações' é 'dividendos'.
    Essa função reduz as chamadas à API agrupando as informações.
    '''
    t = yf.Ticker(ticker)
    def safe_get(attr):
        try:
            data = getattr(t, attr)
            return data if isinstance(data, pd.DataFrame) else pd.DataFrame()
        except Exception as e:
            st.write(f'Erro ao carregar {attr} do ticker {ticker}: {e}')
            return pd.DataFrame()

    financials = safe_get('financials')
    balance_sheet = safe_get('balance_sheet')
    cashflow = safe_get('cashflow')

    df_list = [df for df in [financials, balance_sheet, cashflow] if not df.empty]
    df = pd.concat(df_list, axis=0, sort=False).fillna(0) if df_list else pd.DataFrame()

    # df_info
    try:
        info = t.info
        df_info = pd.DataFrame(info.items(), columns=['Metric', 'Value'])
        df_info.set_index('Metric', inplace=True)
    except Exception as e:
        st.write(f'Erro ao carregar as informações do ticker {ticker}: {e}')
        df_info = pd.DataFrame()

    # df_dividendos
    try:
        df_dividendos = pd.DataFrame(t.dividends, columns=['Dividends'])
        df_dividendos.index.name = 'Date'
    except Exception as e:
        st.write(f'Erro ao carregar os dividendos do ticker {ticker}: {e}')
        df_dividendos = pd.DataFrame()

    return df, df_info, df_dividendos

@st.cache_data(ttl=3600)
def get_selic() -> float:
    url_selic = 'https://www.bcb.gov.br/api/servico/sitebcb/copom/comunicados?quantidade=1'
    try:
        response = requests.get(url_selic, headers={'Accept': 'application/json'}, timeout=10)
        if response.status_code != 200 or not response.text.strip():
            st.write(f'Erro ao obter a taxa SELIC: Código {response.status_code}')
            return 5.00
        selic_json = response.json()
        if selic_json.get('conteudo'):
            titulo = selic_json['conteudo'][0]['titulo']
            selic_rate = float(titulo.split('para')[-1].split('%')[0].replace(',', '.').strip())
        else:
            selic_rate = 5.00
    except (requests.exceptions.RequestException, ValueError, KeyError, IndexError) as e:
        st.write(f'Erro ao processar a taxa SELIC: {e}')
        selic_rate = 5.00
    return selic_rate

# ----------------------- FUNÇÕES DA INTERFACE -----------------------
def analise_individual(ticker_list: list, period_options: dict) -> dict:
    st.subheader('Análise Individual')
    selected_ticker = st.selectbox('Selecione um ativo para análise detalhada:', ticker_list)

    df, df_info, df_dividends = get_yf(selected_ticker)
    df, df_info = df.T, df_info.T

    print(f'\n df: \n{df}\n\n')
    print(f'\n df_info: \n{df_info}\n\n')
    print(f'\n df_dividends: \n{df_dividends}\n\n')

    # Histórico de preços
    data_5y = get_stock_history(selected_ticker, period='5y')
    period_ind = st.selectbox('Selecione o período dos dados históricos:', list(period_options.keys()), key='period_ind')
    data_grafico = get_stock_history(selected_ticker, period=period_options[period_ind])
    if not data_grafico.empty:
        st.write('Histórico de Fechamento:')
        st.line_chart(data_grafico['Close'])
    else:
        st.write('Dados não disponíveis para o período selecionado.')

    # CAGR dos lucros
    try:
        net_income_series = df['Net Income'][df['Net Income'] != 0].dropna()
        if net_income_series.shape[0] >= 2:
            cagr = (net_income_series.iloc[-1] / net_income_series.iloc[0]) ** (1 / (net_income_series.shape[0] - 1)) - 1
        else:
            cagr = 0 
    except Exception as e:
        st.write(f'Erro ao calcular CAGR dos lucros: {e}')
        cagr = 0

    # Fluxo de Caixa
    if 'Free Cash Flow' in df.columns and df['Free Cash Flow'].notna().any():
        cash_flows = df['Free Cash Flow'].dropna().astype(float).values 
    else:
        cash_flows = np.array([])

    # Primeiros dados
    st.subheader('Informações do Ativo')
    st.write(f'**Setor econômico:** {df_info['industry'].iloc[0] if 'industry' in df_info else 'Informação indisponível'}')
    st.write(f'**Setor:** {df_info['setor'].iloc[0] if 'setor' in df_info else 'Informação indisponível'}')
    st.write(f'**Site da Empresa:** {df_info['website'].iloc[0] if 'website' in df_info else 'Informação indisponível'}')

    st.subheader('Indicadores Financeiros')
    df_indicadores = calcular_indicadores(df)
    st.table(df_indicadores.T)

    growth_rate = st.number_input('Taxa de crescimento esperada (%):', value=cagr * 100) / 100
    rf = st.number_input('Taxa de juros (%):', value=get_selic()) / 100
    market_risk_premium = st.number_input('Prêmio pelo risco (%):', value=6) / 100
    tax_rate = st.number_input('Taxa efetiva de imposto (%):', value=30) / 100

    # Calcula o WACC
    try:
        if df.get('Total Debt', 0) != 0 and df.get('Interest Expense', 0) != 0:
            debt_cost = abs(df['Interest Expense'].iloc[0]) / df['Total Debt'].iloc[0]
        else:
            debt_cost = 0.05
    except Exception:
        debt_cost = 0.05

    if 'marketCap' in df_info and 'totalDebt' in df_info:
        market_cap = df_info['marketCap'].iloc[0] if not df_info['marketCap'].empty else 0
        total_debt = df_info['totalDebt'].iloc[0] if not df_info['totalDebt'].empty else 0
        if market_cap != 0:
            total_value = market_cap + total_debt
            equity_ratio = market_cap / total_value if total_value > 0 else 0.5
            debt_ratio = total_debt / total_value if total_value > 0 else 0.5
        else:
            equity_ratio, debt_ratio = 0.5, 0.5
    else:
        equity_ratio, debt_ratio = 0.5, 0.5

    equity_cost = rf + df_info.get('beta', pd.Series([1])).iloc[0] * market_risk_premium
    discount_rate = st.number_input('Taxa de desconto automatica (CAPM): ', 
                                    value=equity_cost * 100) / 100
    
    wacc = (equity_ratio * equity_cost) + (debt_ratio * debt_cost * (1 - tax_rate))

    # Calcula o NPV
    try:
        npv = calculate_npv(cash_flows, wacc) / df_info['sharesOutstanding'].iloc[0] if not df_info['sharesOutstanding'].empty else 0
    except Exception as e:
        st.write(f'Erro ao calcular o NPV: {e}')
        npv = None

    # Calcular Gordon
    df_dividends.index = pd.to_datetime(df_dividends.index)
    df_dividends_sem_y = df_dividends[df_dividends.index.year != pd.Timestamp.today().year]

    if not df_dividends_sem_y.empty:
        div_annualized = df_dividends_sem_y[-4:].sum() if len(df_dividends_sem_y) >= 4 else df_dividends_sem_y.sum()
        div_5y_mean = df_dividends_sem_y.resample('YE').sum().tail(5).mean()

        div_growth = df_dividends_sem_y.resample('YE').sum().tail(7)
        div_growth_pct = div_growth.pct_change().dropna()

        weights = 1 / div_growth_pct.std()
        weights = weights / weights.sum()

        div_growth = (div_growth_pct * weights).sum().item()/100

        fair_price_gordon = gordon_model(div_annualized, div_growth, discount_rate)
        price_5y = gordon_model(div_5y_mean, div_growth, discount_rate)

        # Exibir resultados no Streamlit
        st.write(f'**Dividendo Anualizado:** {div_annualized.iloc[0]:.2f}')
        st.write(f'**Média de Dividendos (5 anos):** {div_5y_mean.iloc[0]:.2f}')
        st.write(f'**Taxa de Crescimento dos Dividendos:** {div_growth:.2%}')
        st.write(f'**Preço Justo (Gordon Anualizado):** {fair_price_gordon:.2f}')
        st.write(f'**Preço Justo (Gordon Média 5 Anos):** {price_5y:.2f}')

    else:
        st.write(f'Nenhum dado de dividendos encontrado para {selected_ticker}')

    # Modelo DCF
    st.subheader('Fluxo de Caixa Descontado (DCF)')
    try:
        if cash_flows.size > 0:
            fair_price_dcf_capm = dcf_model(cash_flows.tolist(), discount_rate, growth_rate)
            fair_price_dcf_wacc = dcf_model(cash_flows.tolist(), wacc, growth_rate)
            shares_outstanding = df_info['sharesOutstanding'].iloc[0] if 'sharesOutstanding' in df_info.columns and not df_info['sharesOutstanding'].empty else None
            
            if shares_outstanding is not None:
                fair_price_dcf = fair_price_dcf_capm / shares_outstanding
                fair_price_dcf_wacc = fair_price_dcf_wacc / shares_outstanding
                st.write(f'Preço justo (DCF com CAPM): {fair_price_dcf:.2f}')
                st.write(f'Preço justo (DCF com WACC): {fair_price_dcf_wacc:.2f}')
            else:
                st.write('Número de ações em circulação indisponível. Não foi possível calcular o preço por ação.')
        else:
            st.write('Fluxo de caixa indisponível para DCF.')
            fair_price_dcf = None

    except Exception as e:
        st.write(f'Erro ao calcular DCF: {e}')
        fair_price_dcf = None

    # Simulação de Preços Futuros
    st.subheader('Previsão de Preços Futuros')
    try:
        hist_30 = data_5y[['Close']].tail(30)
        cenarios_df = gera_cenarios(data_5y[['Close']])
        pess = cenarios_df.quantile(0.1, axis=1)
        med = cenarios_df.quantile(0.5, axis=1)
        opt = cenarios_df.quantile(0.9, axis=1)
        scenario_df = pd.concat([pess, med, opt], axis=1)
        scenario_df.columns = ['Pessimista', 'Mediana', 'Otimista']
        combined_df = pd.concat([hist_30, scenario_df])
        st.line_chart(combined_df)
        future_prices = med
    except Exception as e:
        st.write(f'Erro na previsão de preços: {e}')
        future_prices = None

    current_price = data_5y['Close'].iat[-1] if not data_5y.empty else None

    fair_price = max(fair_price_gordon, fair_price_dcf) if fair_price_gordon and fair_price_dcf else fair_price_gordon or fair_price_dcf

    if all(col in df_info and not df_info[col].empty for col in ['targetLowPrice', 'targetMeanPrice', 'targetHighPrice', 'targetMedianPrice']):
        st.markdown(f'''
        **Previsões do YFinance:**
        - **Mínimo:** {df_info['targetLowPrice'].iloc[0]:.2f}  
        - **Média:** {df_info['targetMeanPrice'].iloc[0]:.2f}  
        - **Máximo:** {df_info['targetHighPrice'].iloc[0]:.2f}  
        - **Mediana:** {df_info['targetMedianPrice'].iloc[0]:.2f}
        ''')
    else:
        st.write('Previsões do YFinance indisponíveis.')
    st.write(f'**NPV (considerando WACC automático de {100*wacc:.2f}%):** {npv:.2f}' if npv is not None else '**NPV:** N/A')
    st.write(f'**Valor Justo:** {fair_price:.2f}' if fair_price is not None else '**Valor Justo:** N/A')
    st.write(f'**Cotação Atual:** {current_price:.2f}' if current_price is not None else '**Cotação Atual:** N/A')
    if future_prices is not None and current_price is not None:
        profit_forecast = ((future_prices.iloc[-1] - current_price) / current_price) * 100
        st.write(f'**Previsão de Lucro:** {profit_forecast:.2f}%')
    else:
        st.write('**Previsão de Lucro:** N/A')
    strength, recommendation = generate_recommendation(fair_price, current_price, future_prices)
    st.write(f'**Força da Oportunidade:** {strength}')
    st.write(f'**Recomendação:** {recommendation}')
    
    return {
        'ticker': selected_ticker,
        'fair_price_gordon': fair_price_gordon,
        'fair_price_dcf': fair_price_dcf,
        'current_price': current_price,
        'future_prices': future_prices,
        'strength': strength,
        'recommendation': recommendation
    }

def comparacao_ativos(ticker_list: list, period_options: dict) -> None:
    st.subheader('Comparação de Ativos')
    comparison_data = pd.DataFrame()
    min_date = None
    selected_period = st.selectbox('Período para comparação:', list(period_options.keys()), key='period_comp')
    for ticker in ticker_list:
        hist_data = get_stock_history(ticker, period=period_options[selected_period])
        if hist_data.empty:
            continue
        hist_data.index = pd.to_datetime(hist_data.index, errors='coerce')
        ticker_min_date = hist_data.index.min()
        if pd.isnull(ticker_min_date):
            continue
        min_date = ticker_min_date if min_date is None else max(min_date, ticker_min_date)
        norm = (hist_data['Close'] / hist_data['Close'].iloc[0]) - 1 if hist_data['Close'].iloc[0] != 0 else 0
        comparison_data[ticker] = norm
    if min_date is not None:
        comparison_data = comparison_data[comparison_data.index >= min_date]
    comparison_data.reset_index(inplace=True)
    melted = comparison_data.melt('Date', var_name='Ticker', value_name='Rendimento')
    melted['Date'] = pd.to_datetime(melted['Date'], errors='coerce')
    selection = alt.selection_point(fields=['Date'], nearest=True)
    chart = alt.Chart(melted).mark_line().encode(
        x='Date:T',
        y=alt.Y('Rendimento:Q', axis=alt.Axis(format='%')),
        color='Ticker:N',
        tooltip=['Date:T', 'Ticker:N', alt.Tooltip('Rendimento:Q', format='.2%')]
    ).add_params(selection)
    points = alt.Chart(melted).mark_point(size=100, opacity=0.008).encode(
        x='Date:T',
        y='Rendimento:Q',
        tooltip=['Date:T', 'Ticker:N', alt.Tooltip('Rendimento:Q', format='.2%')]
    )
    st.altair_chart(chart + points, use_container_width=True)
    
    st.subheader('Análise de Drawdown')
    selected_drawdown_ticker = st.selectbox('Ativo para Drawdown:', ticker_list, key='drawdown_ticker')
    selected_drawdown_period = st.selectbox('Período Drawdown:', list(period_options.keys()), key='drawdown_period')
    drawdown_data = get_stock_history(selected_drawdown_ticker, period=period_options[selected_drawdown_period])
    if not drawdown_data.empty:
        drawdown = (drawdown_data['Close'] / drawdown_data['Close'].cummax() - 1) * 100
        st.line_chart(drawdown)
    else:
        st.write('Dados de drawdown indisponíveis.')
    
    st.subheader('Gráfico de Risco x Retorno Anual')
    risk_return_data = pd.DataFrame()
    for ticker in ticker_list:
        hist = get_stock_history(ticker, period='1y')
        ret = hist['Close'].pct_change().dropna()
        risk = ret.std() * np.sqrt(252)
        mean_ret = ret.mean() * 252
        risk_return_data = pd.concat([risk_return_data, pd.DataFrame([{'Ticker': ticker, 'Risco': risk, 'Retorno': mean_ret}])], ignore_index=True)
    risk_free_rate = st.number_input('Taxa de juros para o gráfico (%):', value=get_selic()) / 100
    risk_return_data = pd.concat([risk_return_data, pd.DataFrame([{'Ticker': 'Risk Free', 'Risco': 0, 'Retorno': risk_free_rate}])], ignore_index=True)
    risk_free_line = alt.Chart(pd.DataFrame({
        'Risco': [0, risk_return_data['Risco'].max()], 
        'Retorno': [risk_free_rate, risk_free_rate]
    })).mark_line(strokeDash=[5, 5], color='red').encode(
        x=alt.X('Risco:Q', axis=alt.Axis(format='.2%', title='Risco (Volatilidade)')),
        y=alt.Y('Retorno:Q', axis=alt.Axis(format='.2%', title='Retorno Médio Anual'))
    )
    risk_return_chart = alt.Chart(risk_return_data).mark_point(size=100).encode(
        x=alt.X('Risco:Q', axis=alt.Axis(format='.2%', title='Risco (Volatilidade)')),
        y=alt.Y('Retorno:Q', axis=alt.Axis(format='.2%', title='Retorno Médio Anual')),
        color='Ticker:N',
        tooltip=['Ticker:N', alt.Tooltip('Risco:Q', format='.2%'), alt.Tooltip('Retorno:Q', format='.2%')]
    ).interactive()
    st.altair_chart(alt.layer(risk_return_chart, risk_free_line), use_container_width=True)

    st.subheader('Gráfico de Correlação')
    selected_corr_period = st.selectbox('Período para correlação:', list(period_options.keys()), key='correlation_period')
    correlation_data = pd.DataFrame()

    for ticker in ticker_list:
        hist_data = get_stock_history(ticker, period=period_options[selected_corr_period])
        if not hist_data.empty:
            correlation_data[ticker] = hist_data['Close'].pct_change()

    if not correlation_data.empty:
        correlation_matrix = correlation_data.corr()
        st.write('Matriz de Correlação:')
        st.dataframe(correlation_matrix)

        # Heatmap de correlação
        fig, ax = plt.subplots(figsize=(10, 8))
        cax = ax.matshow(correlation_matrix, cmap='coolwarm')
        plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
        plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
        fig.colorbar(cax)
        st.pyplot(fig)
    else:
        st.write('Dados insuficientes para calcular a correlação.')

def analise_tecnica(ticker_list: list, period_options: dict) -> None:
    st.subheader('Análise Técnica')
    selected_ticker = st.selectbox('Ativo para análise técnica:', ticker_list, key='tech_ticker')
    tech_period = st.selectbox('Período dos dados:', list(period_options.keys()), key='tech_period')
    tech_data = get_stock_history(selected_ticker, period=period_options[tech_period])
    tech_data['RSI'] = calculate_rsi_manual(tech_data)
    st.write('Índice de Força Relativa (RSI):')
    st.line_chart(tech_data['RSI'])
    
    st.subheader('Média Móvel Simples (SMA)')
    window_short = st.number_input('Período da MMA Curta:', value=50)
    window_long = st.number_input('Período da MMA Longa:', value=200)
    tech_data['SMA_Short'] = tech_data['Close'].rolling(window=window_short).mean()
    tech_data['SMA_Long'] = tech_data['Close'].rolling(window=window_long).mean()
    st.line_chart(tech_data[['Close', 'SMA_Short', 'SMA_Long']])
    
    st.subheader('MACD')
    exp1 = tech_data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = tech_data['Close'].ewm(span=26, adjust=False).mean()
    tech_data['MACD'] = exp1 - exp2
    tech_data['Signal'] = tech_data['MACD'].ewm(span=9, adjust=False).mean()
    st.line_chart(tech_data[['MACD', 'Signal']])
    
    st.subheader('Bandas de Bollinger')
    window_bb = st.number_input('Período das Bandas:', value=20)
    tech_data['Middle Band'] = tech_data['Close'].rolling(window=window_bb).mean()
    tech_data['Upper Band'] = tech_data['Middle Band'] + 2 * tech_data['Close'].rolling(window=window_bb).std()
    tech_data['Lower Band'] = tech_data['Middle Band'] - 2 * tech_data['Close'].rolling(window=window_bb).std()
    st.line_chart(tech_data[['Close', 'Middle Band', 'Upper Band', 'Lower Band']])

def gerar_relatorio(report_content: dict) -> None:
    st.subheader('Gerar Relatório')
    if st.button('Gerar Relatório em PDF'):
        create_pdf_report('relatorio.pdf', report_content)
        st.success('Relatório gerado com sucesso! Verifique o arquivo "relatorio.pdf".')

# ----------------------- INTERFACE STREAMLIT -----------------------
st.title('📈 Analisador Avançado de Ações')
tickers_input = st.text_input('Digite os tickers separados por vírgula (ex: AAPL,VALE3):')
if not tickers_input:
    st.warning('Por favor, insira ao menos um ticker.')
    st.stop()

ticker_list = [
    t.strip().upper() + '.SA' if t.strip()[-1].isdigit() and not t.strip().endswith('.SA') else t.strip().upper()
    for t in tickers_input.split(',') if t.strip()
    ]

if not ticker_list:
    st.error('Nenhum ticker válido foi informado.')
    st.stop()

period_options = {
    'Máximo': 'max',
    '10 anos': '10y',
    '5 anos': '5y',
    '1 ano': '1y',
    '1 mês': '1mo',
    '1 semana': '1wk',
    '1 dia': '1d'
}

tab1, tab2, tab3, tab4 = st.tabs(['Análise Individual', 'Comparação de Ativos', 'Análise Técnica', 'Relatório'])

report_content = {}
with tab1:
    report_content = analise_individual(ticker_list, period_options)
with tab2:
    comparacao_ativos(ticker_list, period_options)
with tab3:
    analise_tecnica(ticker_list, period_options)
with tab4:
    gerar_relatorio(report_content)

# ----------------------- ALERTAS -----------------------
st.sidebar.subheader('Configuração de Alertas')
alert_ticker = st.sidebar.selectbox('Ativo para alerta:', ticker_list)
alert_price = st.sidebar.number_input('Preço de alerta:', value=100.0)
if st.sidebar.button('Ativar Alerta'):
    try:
        current_price_alert = get_stock_history(alert_ticker, period='1d')['Close'].iat[-1]
        if current_price_alert >= alert_price:
            st.sidebar.success(f'Alerta! {alert_ticker} atingiu {current_price_alert:.2f}.')
        else:
            st.sidebar.info(f'Aguardando {alert_ticker} atingir {alert_price:.2f}.')
    except Exception as e:
        st.sidebar.error(f'Erro ao obter o preço atual: {e}')