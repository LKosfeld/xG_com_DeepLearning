#Carregamento dos dados----
library(tidyverse)
library(keras)
todas_ligas_preparados <- read.csv("dados/preparados/todas_ligas_preparados.csv")

#1. PREPARAÇÃO DOS DADOS----
# (Esta seção está correta e permanece inalterada)
features_df <- todas_ligas_preparados %>% select(-is_goal)
labels_vector <- todas_ligas_preparados$is_goal
features_matrix <- model.matrix(~ . -1, data = features_df)
set.seed(123)
train_indices <- sample(1:nrow(features_matrix), size = 0.8 * nrow(features_matrix))
train_data <- features_matrix[train_indices, ]
train_labels <- labels_vector[train_indices]
test_data <- features_matrix[-train_indices, ]
test_labels <- labels_vector[-train_indices]


#2. CONSTRUÇÃO DO MODELO----
# (Esta seção está correta e permanece inalterada)
build_model <- function(learn_rate = 0.001, dropout_rate = 0.2, l2_reg = 0.01, units1 = 64, units2 = 32) {
  model <- keras_model_sequential() %>%
    layer_dense(units = units1, activation = "relu",
                kernel_regularizer = regularizer_l2(l2_reg), 
                input_shape = c(ncol(train_data))) %>%
    layer_dropout(rate = dropout_rate) %>% 
    layer_dense(units = units2, activation = "relu",
                kernel_regularizer = regularizer_l2(l2_reg)) %>% 
    layer_dropout(rate = dropout_rate) %>% 
    layer_dense(units = 1, activation = "sigmoid")
  
  model %>% compile(
    loss = "binary_crossentropy",
    optimizer = optimizer = optimizer_adam(learning_rate = learn_rate),
    metrics = c("accuracy", "AUC")
  )
  
  return(model)
}

#3. Definição do grid de hiperparametros
hyper_grid_completo <- expand.grid(
  learn_rate = c(0.001, 0.0005),
  dropout_rate = c(0.2, 0.3, 0.4, 0.5),
  l2_reg = c(0.1, 0.01, 0.005, 0.001),
  units1 = c(32, 64, 128, 256),
  units2 = c(32, 64, 128, 256),
  stringsAsFactors = FALSE
)

hyper_grid_completo <- hyper_grid_completo %>%
  filter(units2 <= units1)

# ---- LÓGICA CORRIGIDA PARA "PARAR E CONTINUAR" ----

arquivo_progresso <- "resultados_parciais_grid_search.csv"

if (file.exists(arquivo_progresso)) {
  
  cat("Arquivo de progresso encontrado! Carregando resultados anteriores...\n")
  grid_search_results <- read.csv(arquivo_progresso)
  
  # CORREÇÃO: Compara o grid completo com os resultados salvos para encontrar o que falta
  hyper_grid_a_fazer <- anti_join(hyper_grid_completo, grid_search_results, by = names(hyper_grid_completo))
  
  cat(nrow(grid_search_results), "combinações já foram testadas.\n")
  cat("Restam", nrow(hyper_grid_a_fazer), "combinações para testar.\n\n")
  
} else {
  
  cat("Nenhum progresso anterior encontrado. Iniciando do zero...\n\n")
  # CORREÇÃO: Inicializa o dataframe de resultados vazio APENAS se o arquivo não existir
  grid_search_results <- data.frame() 
  hyper_grid_a_fazer <- hyper_grid_completo 
  
}

#4. Otimização com VALIDAÇÃO CRUZADA K-FOLD----

k <- 4
num_epochs <- 150

# CORREÇÃO: O loop deve iterar sobre a lista de tarefas PENDENTES (hyper_grid_a_fazer)
for (i in 1:nrow(hyper_grid_a_fazer)) {
  
  # CORREÇÃO: Pega os parâmetros da lista de tarefas pendentes
  params <- hyper_grid_a_fazer[i, ]
  
  cat("---------------------------------------------------\n")
  cat("Processando combinação", i, "/", nrow(hyper_grid_a_fazer), 
      "(Total Concluído:", nrow(grid_search_results), "de", nrow(hyper_grid_completo), ")\n")
  print(as.data.frame(params))
  
  all_val_auc_histories <- list()
  all_epochs_ran <- c()
  
  indices <- sample(1:nrow(train_data))
  folds <- cut(indices, breaks = k, labels = FALSE)
  
  for (j in 1:k) {
    val_indices <- which(folds == j, arr.ind = TRUE)
    
    val_data <- train_data[val_indices, ]
    val_labels <- train_labels[val_indices]
    partial_train_data <- train_data[-val_indices, ]
    partial_train_labels <- train_labels[-val_indices]
    
    train_mean <- colMeans(partial_train_data)
    train_std <- apply(partial_train_data, 2, sd)
    partial_train_data <- scale(partial_train_data, center = train_mean, scale = train_std)
    val_data <- scale(val_data, center = train_mean, scale = train_std)
    
    model <- build_model(learn_rate = params$learn_rate, 
                         dropout_rate = params$dropout_rate, 
                         l2_reg = params$l2_reg,
                         units1 = params$units1,
                         units2 = params$units2)
    
    early_stop <- callback_early_stopping(monitor = "val_loss", patience = 15, restore_best_weights = TRUE)
    
    history <- model %>% fit(
      partial_train_data, partial_train_labels,
      validation_data = list(val_data, val_labels),
      epochs = num_epochs, 
      batch_size = 16, 
      verbose = 0,
      callbacks = list(early_stop)
    )
    
    all_val_auc_histories[[j]] <- history$metrics$val_auc
    epochs_ran_this_fold <- max(1, length(history$metrics$val_loss) - 15)
    all_epochs_ran <- c(all_epochs_ran, epochs_ran_this_fold)
  }
  
  mean_final_val_auc <- mean(sapply(all_val_auc_histories, tail, 1))
  
  params$mean_auc <- mean_final_val_auc
  params$mean_epochs <- ceiling(mean(all_epochs_ran))
  
  # CORREÇÃO: Adiciona a nova linha de resultado ao dataframe CUMULATIVO
  grid_search_results <- bind_rows(grid_search_results, params)
  
  cat("  --> AUC Média de Validação:", round(mean_final_val_auc, 4), "\n")
  
  # SALVA o dataframe CUMULATIVO e ATUALIZADO no arquivo.
  write.csv(grid_search_results, arquivo_progresso, row.names = FALSE)
  cat("  --> Progresso salvo no arquivo '", arquivo_progresso, "'\n\n")
  
  k_clear_session()
}

xat("\n--- OTIMIZAÇÃO CONCLUÍDA ---\n")

# ... O resto do seu script para encontrar os melhores parâmetros e treinar o modelo final continua a partir daqui ...

#5. TREINAMENTO DO MODELO FINAL----
cat("\nIniciando o treinamento do modelo final...\n")

# CORREÇÃO: Use TODOS os melhores parâmetros para construir o modelo final
model_final <- build_model(learn_rate = best_params$learn_rate,
                           dropout_rate = best_params$dropout_rate,
                           l2_reg = best_params$l2_reg,
                           units1 = best_params$units1,
                           units2 = best_params$units2)

final_mean <- colMeans(train_data)
final_std <- apply(train_data, 2, sd)
train_data_scaled <- scale(train_data, center = final_mean, scale = final_std)
test_data_scaled <- scale(test_data, center = final_mean, scale = final_std)

history_final <- model_final %>% fit(
  train_data_scaled, train_labels,
  epochs = best_params$mean_epochs,
  batch_size = 16, 
  verbose = 1
)

#6. AVALIAÇÃO FINAL NO CONJUNTO DE TESTE----
results <- model_final %>% evaluate(test_data_scaled, test_labels)
print(results)