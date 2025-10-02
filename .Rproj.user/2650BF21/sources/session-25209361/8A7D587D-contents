obter_chutes <- function(competicao_df, 
                         salvar_resultado = FALSE, 
                         nome_arquivo = "dados_chutes.csv", 
                         caminho_pasta = ".") {
  
  # --- 1. Processamento dos dados
  message("Processando: ", competicao_df$competition_name[1], " (Season: ", competicao_df$season_name[1], ")")
  
  matches <- FreeMatches(Competitions = competicao_df)
  events <- free_allevents(MatchesDF = matches, Parallel = TRUE)
  events <- allclean(events)
  
  shots <- events %>%
    filter(type.name == "Shot")
  
  
  # --- 2. Lógica para salvar o resultado
  if (salvar_resultado == TRUE) {
    
    message("Opção 'salvar_resultado' ativada. Salvando arquivo...")
    
    # Criar a pasta se ela não existir
    if (!dir.exists(caminho_pasta)) {
      message("Criando a pasta: '", caminho_pasta, "'")
      dir.create(caminho_pasta, recursive = TRUE)
    }
    
    # Garantir que o nome do arquivo termina com .csv
    if (!grepl("\\.csv$", nome_arquivo, ignore.case = TRUE)) {
      nome_arquivo <- paste0(nome_arquivo, ".csv")
    }
    
    # Montar o caminho completo do arquivo
    caminho_completo <- file.path(caminho_pasta, nome_arquivo)
    
    # Salvar o arquivo usando write_csv
    readr::write_csv(shots, caminho_completo)
    
    # Mensagem de confirmação
    message("Arquivo salvo com sucesso em: \n", caminho_completo)
  }
  
  # --- 3. Retornar o data frame (sempre) ---
  return(shots)
}


#Função para preparar os dados para o modelo de xG
preparar_dados_xg <- function(dataframe,
                              salvar_resultado = FALSE,
                              nome_arquivo = "dados_processados.csv",
                              caminho_pasta = "dados_processados") {
  
  # --- 1. Processamento dos dados ---
  message("Processando e selecionando features...")
  
  dados_processados <- dataframe %>%
    # Cria a variável alvo (label)
    mutate(is_goal = ifelse(shot.outcome.name == "Goal", 1, 0)) %>%
    # Seleciona as features
    select(
      is_goal, play_pattern.name, position.name, shot.type.name,
      shot.technique.name, shot.body_part.name, location.x,
      location.y, location.x.GK, location.y.GK, DistToGoal,
      DistToKeeper, AngleToGoal, AngleToKeeper, AngleDeviation,
      avevelocity, DistSGK, density, density.incone,
      distance.ToD1, distance.ToD2, AttackersBehindBall,
      DefendersBehindBall, DefendersInCone, InCone.GK, DefArea
    ) %>%
    na.omit()
  
  message("Processamento concluído. Total de observações limpas:", nrow(dados_processados))
  
  # --- 2. Lógica para salvar o resultado (similar à sua função) ---
  if (salvar_resultado == TRUE) {
    
    message("Opção 'salvar_resultado' ativada. Salvando arquivo...")
    
    # Criar a pasta se ela não existir
    if (!dir.exists(caminho_pasta)) {
      message("Criando a pasta: '", caminho_pasta, "'")
      dir.create(caminho_pasta, recursive = TRUE)
    }
    
    # Garantir que o nome do arquivo termina com .csv
    if (!grepl("\\.csv$", nome_arquivo, ignore.case = TRUE)) {
      nome_arquivo <- paste0(nome_arquivo, ".csv")
    }
    
    # Montar o caminho completo do arquivo
    caminho_completo <- file.path(caminho_pasta, nome_arquivo)
    
    # Salvar o arquivo usando write_csv para consistência
    readr::write_csv(dados_processados, caminho_completo)
    
    # Mensagem de confirmação
    message("Arquivo salvo com sucesso em: \n", caminho_completo)
  }
  
  # --- 3. Retornar o data frame (sempre) ---
  return(dados_processados)
}
