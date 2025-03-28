---

# Análise Financeira de Ações

Este repositório contém um conjunto de funções e modelos projetados para realizar análises financeiras detalhadas de ações, com foco em indicadores fundamentais, avaliação de preço justo e análise técnica. As funções são integradas em um ambiente interativo no **Streamlit**, facilitando a visualização e interpretação dos resultados.

### Funcionalidades

O projeto está dividido em quatro seções principais, organizadas em tabelas no Streamlit:

## 1. **Indicadores Financeiros Fundamentais**

A primeira tabela calcula diversos **indicadores financeiros**, fundamentais para avaliar a saúde financeira de uma empresa e sua performance no mercado. Os indicadores incluem:

- **Liquidez:** Mede a capacidade da empresa de honrar suas obrigações de curto prazo.
- **Solvência:** Avalia a capacidade da empresa de se manter financeiramente no longo prazo.
- **Endividamento:** Relaciona o nível de dívidas da empresa com sua estrutura de capital.
- **Cobertura de Juros:** Indica a capacidade da empresa de pagar juros sobre sua dívida.
- **Multiplicadores Financeiros:** Mede a relação entre variáveis financeiras, como preço sobre lucro (P/L), preço sobre valor patrimonial (P/VPA), entre outros.
- **Margens e Rentabilidade:** Avalia as margens de lucro e a rentabilidade da empresa.

Além dos indicadores, esta seção também tenta aproximar o cálculo de um **preço justo** para o ativo, utilizando os seguintes dados calculados automaticamente:

- **CAGR (Taxa de Crescimento Anual Composta):** Calcula o crescimento médio anual de uma métrica específica ao longo do tempo.
- **WACC (Custo Médio Ponderado de Capital):** Taxa de retorno exigida pelos investidores.
- **CAPM (Modelo de Precificação de Ativos Financeiros):** Estima o retorno esperado de um ativo com base no risco sistemático.
- **Taxa SELIC:** Utilizada como taxa livre de risco no Brasil.

Esses dados alimentam modelos como o **Modelo de Gordon**, **NPV (Valor Presente Líquido)** e o **Modelo de Fluxo de Caixa Descontado (DCF)**, proporcionando uma avaliação robusta do preço justo da ação. Além disso, a função simula **cenários futuros** com base em **retornos históricos**, permitindo ao usuário visualizar possíveis variações no preço da ação.

## 2. **Comparação entre Ativos**

A segunda tabela permite comparar múltiplos ativos, levando em consideração:

- **Rendimento:** A rentabilidade histórica de cada ativo.
- **Drawdown:** A maior queda no valor de um ativo em um determinado período.
- **Risco x Retorno:** A relação entre o risco (volatilidade) e o retorno obtido.
- **Correlação:** A correlação entre os ativos, ajudando na análise de diversificação.

Essa análise facilita a escolha de ativos para a construção de portfólios mais eficientes e balanceados.

## 3. **Análise Técnica**

A terceira tabela apresenta uma análise técnica com indicadores populares para o acompanhamento de ações:

- **RSI (Índice de Força Relativa):** Indica se um ativo está sobrecomprado ou sobrevendido, ajudando na identificação de potenciais pontos de reversão de tendência.
- **Média Móvel:** Utilizada para suavizar as flutuações de preços e identificar tendências.
- **MACD (Convergência e Divergência de Médias Móveis):** Um indicador de momentum que ajuda a identificar mudanças na força, direção, momento e duração de uma tendência de preço.
- **Bandas de Bollinger:** Indicador de volatilidade que ajuda a identificar condições de sobrecompra ou sobrevenda.

Esses indicadores auxiliam na análise do comportamento do preço da ação no curto prazo, com foco em sinais de compra ou venda.

## 4. **Geração de Relatório**

A última tabela permite a **geração de relatórios em PDF**, que incluem todos os resultados das análises financeiras realizadas. O relatório é gerado automaticamente com gráficos e dados detalhados, oferecendo uma visão completa do ativo analisado, além de facilitar a documentação e apresentação dos resultados.


## Como Usar

1. **Instale as bibliotecas necessárias**:

   Primeiramente, crie um ambiente virtual (opcional, mas recomendado) e instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

   Se não houver um arquivo `requirements.txt`, você pode instalar as bibliotecas manualmente, como:

   ```bash
   pip install pandas numpy streamlit
   ```

2. **Importe as funções do repositório**:

   No seu script Python, importe as funções conforme necessário. Exemplo:

   ```python
   from nome_do_repositorio import calculate_rsi_manual, gordon_model, dcf_model, gera_cenarios, etc.
   ```

3. **Execute a aplicação Streamlit**:

   Para rodar a interface interativa com Streamlit, use o seguinte comando:

   ```bash
   streamlit run app.py
   ```

   Isso abrirá a aplicação no seu navegador, onde você poderá visualizar os resultados de suas análises financeiras interativas.

---

### Detalhes adicionais

- **Personalização**: Altere os parâmetros de entrada (como dividendos, taxa de crescimento, etc.) para ajustar as funções conforme a sua necessidade.
- **Cenários Futuros**: Use a função `gera_cenarios` para simular diferentes cenários de preços com base nos dados históricos.
- **Relatório**: Aplique as funções para gerar um relatório detalhado em PDF.

---
## Contribuindo

Se você deseja contribuir com este projeto, sinta-se à vontade para fazer um fork, abrir pull requests e compartilhar melhorias com a comunidade!

## Licença

Este projeto está licenciado sob a **Licença MIT**. Consulte o arquivo [LICENSE](LICENSE) para mais detalhes.

---
