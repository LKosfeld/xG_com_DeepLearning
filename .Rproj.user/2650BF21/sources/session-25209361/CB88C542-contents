#setup----
#install.packages("devtools")
#devtools::install_github("statsbomb/StatsBombR")
library(StatsBombR)
library(tidyverse)
library(dplyr)
source("funcoes.R")

#carregamento de dados----
## !!RODAR APENAS 1 VEZ, DEPOIS O CSV FICARÁ SALVO!!
##bundesliga
bundesliga <- FreeCompetitions() %>%
  filter(season_id == 27, competition_id == 9)
chutes_bundesliga <- obter_chutes(bundesliga, TRUE, "chutes_bundesliga.csv", "dados/brutos")
##la liga
la_liga <- FreeCompetitions() %>%
  filter(season_id == 27, competition_id == 11)
chutes_la_liga <- obter_chutes(la_liga, TRUE, "chutes_la_liga.csv", "dados/brutos")
##ligue 1
ligue1 <- FreeCompetitions() %>%
  filter(season_id == 27, competition_id == 7)
chutes_ligue <- obter_chutes(ligue1, TRUE, "chutes_ligue1.csv", "dados/brutos")
##premier league
premier <- FreeCompetitions() %>%
  filter(season_id == 27, competition_id == 2)
chutes_premier <- obter_chutes(premier, TRUE, "chutes_premier.csv", "dados/brutos")
##serie a
serie_a <- FreeCompetitions() %>%
  filter(season_id == 27, competition_id == 12)
chutes_serie_a <- obter_chutes(serie_a, TRUE, "chutes_serie_a.csv", "dados/brutos")

#Seleção de Features para usar no modelo final----
bundesliga_preparados <- preparar_dados_xg(chutes_bundesliga, TRUE, "bundesliga_preparados.csv", "dados/preparados")
la_liga_preparados <- preparar_dados_xg(chutes_la_liga, TRUE, "la_liga_preparados.csv", "dados/preparados")
ligue1_preparados <- preparar_dados_xg(chutes_ligue1, TRUE, "ligue1_preparados.csv", "dados/preparados")
premier_preparados <- preparar_dados_xg(chutes_premier, TRUE, "premier_preparados.csv", "dados/preparados")
serie_a_preparados <- preparar_dados_xg(chutes_serie_a, TRUE, "serie_a_preparados.csv", "dados/preparados")
todos_chutes <- bind_rows(chutes_bundesliga, chutes_la_liga,
                      chutes_ligue1, chutes_premier, chutes_serie_a)
todas_ligas_preparados <- preparar_dados_xg(todos_chutes, TRUE, "todas_ligas_preparados.csv", "dados/preparados")


