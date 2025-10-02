#Carregar os dados salvos no script anterior----
chutes_bundesliga <- read.csv("dados/chutes_bundesliga.csv")
chutes_la_liga <- read.csv("dados/chutes_la_liga.csv")
chutes_ligue1 <- read.csv("dados/chutes_ligue1.csv")
chutes_premier <- read.csv("dados/chutes_premier.csv")
chutes_serie_a <- read.csv("dados/chutes_serie_a.csv")

#Graficos e analise basicas----
library(tidyverse)

##Contar os chutes por liga
num_chutes_bundesliga <- nrow(chutes_bundesliga)
num_chutes_la_liga <- nrow(chutes_la_liga)
num_chutes_ligue1 <- nrow(chutes_ligue1)
num_chutes_premier <- nrow(chutes_premier)
num_chutes_serie_a <- nrow(chutes_serie_a)

##Novo DF com os chutes por liga
dados_grafico <- data.frame(
  Liga = c("Bundesliga", "La Liga", "Ligue 1", "Premier League", "Serie A"),
  QuantidadeDeChutes = c(num_chutes_bundesliga, num_chutes_la_liga, num_chutes_ligue1, num_chutes_premier, num_chutes_serie_a)
)

##Grafico de chutes por liga
ggplot(dados_grafico, aes(x = reorder(Liga, -QuantidadeDeChutes), y = QuantidadeDeChutes)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  geom_text(aes(label = QuantidadeDeChutes), vjust = -0.3, size = 3.5) +
  labs(
    title = "Quantidade Total de Chutes por Liga",
    subtitle = "Comparativo entre as 5 principais ligas da Europa",
    x = "Liga",
    y = "Número de Chutes"
  ) +
  theme_minimal()

##Novo DF para todas as liga
todos_os_chutes <- bind_rows(
  chutes_bundesliga %>% mutate(Liga = "Bundesliga"),
  chutes_la_liga    %>% mutate(Liga = "La Liga"),
  chutes_ligue1     %>% mutate(Liga = "Ligue 1"),
  chutes_premier    %>% mutate(Liga = "Premier League"),
  chutes_serie_a    %>% mutate(Liga = "Serie A")
)

analise_gols <- todos_os_chutes %>%
  group_by(Liga) %>%
  summarise(
    TotalChutes = n(),
    Gols = sum(shot.outcome.name == "Goal", na.rm = TRUE),
    TaxaConversao = Gols / TotalChutes
  ) %>%
  arrange(desc(Gols))

##Grafico de Gols por Liga
ggplot(analise_gols, aes(x = reorder(Liga, -Gols), y = Gols)) +
  geom_bar(stat = "identity", fill = "#2ECC71") +
  geom_text(aes(label = Gols), vjust = -0.3, size = 3.5) +
  labs(
    title = "Número de Gols Marcados por Liga",
    x = "Liga",
    y = "Total de Gols"
  ) +
  theme_minimal()

##Grafico de taxa de conversao
ggplot(analise_gols, aes(x = reorder(Liga, -TaxaConversao), y = TaxaConversao)) +
  geom_bar(stat = "identity", fill = "#3498DB") +
  
  # Usamos a função scales::percent() para formatar o rótulo como porcentagem
  geom_text(aes(label = scales::percent(TaxaConversao, accuracy = 0.1)), vjust = -0.3, size = 3.5) +
  
  # Formatamos o eixo Y para também mostrar porcentagens
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  
  labs(
    title = "Taxa de Conversão de Chutes em Gol (%)",
    subtitle = "Eficiência de finalização por liga",
    x = "Liga",
    y = "Taxa de Conversão"
  ) +
  theme_minimal()

##Gols por jogo
analise_gols_por_jogo <- todos_os_chutes %>%
  group_by(Liga) %>%
  summarise(
    Gols = sum(shot.outcome.name == "Goal", na.rm = TRUE),
    Jogos = n_distinct(match_id),
    GolsPorJogo = Gols / Jogos
  ) %>%
  arrange(desc(GolsPorJogo))

ggplot(analise_gols_por_jogo, aes(x = reorder(Liga, -GolsPorJogo), y = GolsPorJogo)) +
  geom_bar(stat = "identity", fill = "#E74C3C") +
  
  # Adiciona o rótulo com a média de gols, formatado para 2 casas decimais
  geom_text(aes(label = round(GolsPorJogo, 2)), vjust = -0.3, size = 3.5) +
  
  # Ajusta a escala do eixo Y para dar um pouco de espaço acima das barras
  scale_y_continuous(expand = expansion(mult = c(0, 0.1))) +
  
  labs(
    title = "Média de Gols por Jogo em Cada Liga",
    subtitle = "Qual liga tem os jogos com mais gols?",
    x = "Liga",
    y = "Média de Gols por Jogo"
  ) +
  theme_minimal()
n_distinct(chutes_ligue1$match_id)
