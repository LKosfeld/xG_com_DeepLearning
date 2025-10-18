#Carregamento dos dados----
library(tidyverse)
library(keras)
library(pROC)
library(PRROC)
library(caret)

todas_ligas_preparados <- read.csv("dados/preparados/todas_ligas_preparados.csv")

#1. PREPARAÇÃO DOS DADOS----

features_df <- todas_ligas_preparados %>% select(-is_goal)
labels_vector <- todas_ligas_preparados$is_goal

##Transforma features categóricas em numéricas (one-hot encoding) e cria a matriz
features_matrix <- model.matrix(~ . -1, data = features_df)

##Divisão em conjuntos de treino (80%) e teste (20%)
set.seed(17)
train_indices <- sample(1:nrow(features_matrix), size = 0.8 * nrow(features_matrix))

train_data <- features_matrix[train_indices, ]
train_labels <- labels_vector[train_indices]
test_data <- features_matrix[-train_indices, ]
test_labels <- labels_vector[-train_indices]


#2. CONSTRUÇÃO DO MODELO----
build_model <- function() {
  model <- keras_model_sequential() %>%
    ##Camada de entrada com regularização L2
    layer_dense(units = 64, activation = "relu",
                kernel_regularizer = regularizer_l2(0.005), 
                input_shape = c(ncol(train_data))) %>%
    ##Camada de Dropout
    layer_dropout(rate = 0.3) %>% 
    
    ##Camada oculta com regularização L2
    layer_dense(units = 32, activation = "relu",
                kernel_regularizer = regularizer_l2(0.005)) %>% 
    ##Outra camada de Dropout
    layer_dropout(rate = 0.3) %>% 
    
    ##Camada de saída com ativação sigmoid para probabilidade
    layer_dense(units = 1, activation = "sigmoid")
  
  model %>% compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_adam(learning_rate = 0.0005),
    metrics = c("accuracy", "AUC")
  )
  
  return(model)
}


#3. VALIDAÇÃO CRUZADA K-FOLD----

k <- 4
num_epochs <- 150

indices <- sample(1:nrow(train_data))
folds <- cut(indices, breaks = k, labels = FALSE)

all_val_auc_histories <- list()
all_epochs_ran <- c()

# CALLBACKS MELHORADOS PARA EARLY STOPPING
improved_callbacks <- list(
  callback_early_stopping(
    monitor = "val_auc",           # Monitorar AUC (mais importante para xG)
    patience = 10,                 # Esperar 10 épocas sem melhoria
    restore_best_weights = TRUE,   # Restaurar pesos da melhor época
    mode = "max",                  # Queremos maximizar a AUC
    verbose = 1
  ),
  callback_reduce_lr_on_plateau(
    monitor = "val_loss",
    factor = 0.5,                  # Reduz learning rate pela metade
    patience = 5,                  # Espera 5 épocas
    min_lr = 0.00001,
    verbose = 1
  )
)

for (i in 1:k) {
  cat("Processando fold de validação #", i, "de", k, "\n")
  val_indices <- which(folds == i, arr.ind = TRUE)
  
  ##Dados de validação e treino parcial para este fold
  val_data <- train_data[val_indices, ]
  val_labels <- train_labels[val_indices]
  partial_train_data <- train_data[-val_indices, ]
  partial_train_labels <- train_labels[-val_indices]
  
  ##Normalização dentro do loop
  train_mean <- colMeans(partial_train_data)
  train_std <- apply(partial_train_data, 2, sd)
  partial_train_data <- scale(partial_train_data, center = train_mean, scale = train_std)
  val_data <- scale(val_data, center = train_mean, scale = train_std)
  
  ##Construção e Treinamento do modelo
  model <- build_model()
  
  ## USANDO CALLBACKS MELHORADOS
  history <- model %>% fit(
    partial_train_data, partial_train_labels,
    validation_data = list(val_data, val_labels),
    epochs = num_epochs, 
    batch_size = 16, 
    verbose = 0,
    callbacks = improved_callbacks,  # Callbacks melhorados
  )
  
  all_val_auc_histories[[i]] <- history$metrics$val_auc
  epochs_ran <- length(history$metrics$val_loss) #Número de épocas que realmente rodaram
  all_epochs_ran <- c(all_epochs_ran, epochs_ran)
  
  cat("  Fold", i, "- AUC de Validação Final:", round(tail(history$metrics$val_auc, 1), 4),
      "- Treinou por:", epochs_ran, "épocas.\n")
}

##Análise dos resultados da validação cruzada
mean_final_val_auc <- mean(sapply(all_val_auc_histories, tail, 1))
ideal_epochs <- ceiling(mean(all_epochs_ran)) #Define o n° de épocas ideal com base na média

cat("\n--- Resultados da Validação Cruzada K-Fold ---\n")
cat("AUC média final de validação:", round(mean_final_val_auc, 4), "\n")
cat("Número médio de épocas sugerido para o treino final:", ideal_epochs, "\n\n")



#4. TREINAMENTO DO MODELO FINAL----

cat("\nIniciando o treinamento do modelo final...\n")
model_final <- build_model()

##Normalizar os dados de treino e teste com base APENAS nos dados de treino
final_mean <- colMeans(train_data)
final_std <- apply(train_data, 2, sd)
train_data_scaled <- scale(train_data, center = final_mean, scale = final_std)
test_data_scaled <- scale(test_data, center = final_mean, scale = final_std)

##Treino final COM CALLBACKS MELHORADOS
history_final <- model_final %>% fit(
  train_data_scaled, train_labels,
  epochs = ideal_epochs,
  batch_size = 16, 
  verbose = 1,
  validation_split = 0.2,  # Adiciona validação durante treino final
  callbacks = improved_callbacks  # Usando callbacks melhorados
)



#5. AVALIAÇÃO FINAL NO CONJUNTO DE TESTE----
results <- model_final %>% evaluate(test_data_scaled, test_labels)
cat("\n=== RESULTADOS NO CONJUNTO DE TESTE ===\n")
print(results)



#6. COMPARAÇÃO DE CALIBRAÇÃO: GOLS REAIS vs. GOLS ESPERADOS (xG)----

cat("\n--- Análise de Calibração Agregada ---\n")

##Fazer as previsões de probabilidade no conjunto de teste
predictions <- model_final %>% predict(test_data_scaled)

##A saída de predict() é uma matriz, então convertemos para um vetor numérico
predicted_probs <- as.vector(predictions)

##Calcular a soma dos gols que realmente aconteceram no conjunto de teste
gols_reais <- sum(test_labels)

##Calcular a soma de todas as probabilidades previstas (o xG Total)
gols_esperados <- sum(predicted_probs)

##Apresentar os resultados para comparação
cat("Total de Gols Reais no conjunto de teste: ", gols_reais, "\n")
cat("Total de Gols Esperados (soma do xG) pelo modelo:", round(gols_esperados, 2), "\n")

##Adicional: Calcular a diferença percentual
diferenca_percentual <- ((gols_esperados - gols_reais) / gols_reais) * 100
cat("Diferença Percentual: ", round(diferenca_percentual, 2), "%\n\n")



#7. NOVAS MÉTRICAS PARA AVALIAÇÃO DO MODELO xG----

cat("\n=== MÉTRICAS DETALHADAS DO MODELO xG ===\n")

# Função para calcular métricas adicionais
calculate_xg_metrics <- function(probs, labels) {
  
  # Brier Score (quanto menor melhor)
  brier_score <- mean((probs - labels)^2)
  
  # Log Loss (com proteção contra log(0))
  epsilon <- 1e-15
  probs_clipped <- pmin(pmax(probs, epsilon), 1-epsilon)
  log_loss <- -mean(labels * log(probs_clipped) + (1-labels) * log(1-probs_clipped))
  
  # AUC-ROC
  roc_curve <- roc(labels, probs)
  auc_roc <- auc(roc_curve)
  
  # AUC-PR (Precision-Recall - importante para classes desbalanceadas)
  pr_curve <- pr.curve(scores.class0 = probs[labels == 1], 
                       scores.class1 = probs[labels == 0])
  auc_pr <- pr_curve$auc.integral
  
  # Métricas por faixa de xG
  high_xg_metrics <- data.frame(
    threshold = c(0.1, 0.2, 0.3, 0.4, 0.5),
    precision = sapply(c(0.1, 0.2, 0.3, 0.4, 0.5), function(t) {
      mean(labels[probs > t])
    }),
    recall = sapply(c(0.1, 0.2, 0.3, 0.4, 0.5), function(t) {
      sum(labels[probs > t]) / sum(labels)
    }),
    n_shots = sapply(c(0.1, 0.2, 0.3, 0.4, 0.5), function(t) {
      sum(probs > t)
    })
  )
  
  return(list(
    brier_score = brier_score,
    log_loss = log_loss,
    auc_roc = as.numeric(auc_roc),
    auc_pr = auc_pr,
    high_xg_metrics = high_xg_metrics
  ))
}

# Calcular métricas detalhadas
xg_metrics <- calculate_xg_metrics(predicted_probs, test_labels)

cat("--- Métricas de Desempenho ---\n")
cat("Brier Score:", round(xg_metrics$brier_score, 4), "(quanto menor melhor)\n")
cat("Log Loss:", round(xg_metrics$log_loss, 4), "(quanto menor melhor)\n")
cat("AUC-ROC:", round(xg_metrics$auc_roc, 4), "\n")
cat("AUC-PR:", round(xg_metrics$auc_pr, 4), "(importante para classes desbalanceadas)\n")

cat("\n--- Desempenho em Finalizações de Alta Qualidade ---\n")
print(xg_metrics$high_xg_metrics)



#8. ANÁLISE DE CALIBRAÇÃO DETALHADA----

cat("\n=== ANÁLISE DETALHADA DE CALIBRAÇÃO ===\n")

calibration_analysis_detailed <- function(probs, labels, bins = 10) {
  # Criar bins
  breaks <- seq(0, 1, length.out = bins + 1)
  bin_midpoints <- (breaks[-1] + breaks[-(bins+1)]) / 2
  
  # Calcular estatísticas por bin
  binned <- cut(probs, breaks = breaks, include.lowest = TRUE)
  
  calibration_df <- data.frame(
    bin = levels(binned),
    bin_midpoint = bin_midpoints,
    n_observations = as.numeric(table(binned)),
    predicted_prob = tapply(probs, binned, mean),
    actual_prob = tapply(labels, binned, mean)
  )
  
  # Calcular estatísticas de calibração
  calibration_df <- calibration_df %>%
    mutate(
      difference = actual_prob - predicted_prob,
      abs_difference = abs(difference),
      expected_goals = predicted_prob * n_observations,
      actual_goals = actual_prob * n_observations
    )
  
  return(calibration_df)
}

# Executar análise de calibração detalhada
calib_results <- calibration_analysis_detailed(predicted_probs, test_labels)

# Estatísticas de calibração
calibration_stats <- calib_results %>%
  summarise(
    ECE = weighted.mean(abs_difference, n_observations),  # Expected Calibration Error
    MCE = max(abs_difference),                            # Maximum Calibration Error
    correlation = cor(predicted_prob, actual_prob),
    bias = mean(difference)
  )

cat("--- Estatísticas de Calibração ---\n")
cat("Expected Calibration Error (ECE):", round(calibration_stats$ECE, 4), "\n")
cat("Maximum Calibration Error (MCE):", round(calibration_stats$MCE, 4), "\n")
cat("Correlação Previsto vs Real:", round(calibration_stats$correlation, 4), "\n")
cat("Bias Médio:", round(calibration_stats$bias, 4), "\n")

# Plotar curva de calibração
plot_calibration_curve <- function(calib_df) {
  ggplot(calib_df, aes(x = predicted_prob, y = actual_prob)) +
    geom_point(aes(size = n_observations), color = "steelblue", alpha = 0.7) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
    geom_smooth(method = "lm", se = FALSE, color = "darkblue") +
    labs(
      title = "Curva de Calibração do Modelo xG",
      x = "Probabilidade Prevista (xG)",
      y = "Frequência Real de Gol",
      size = "N° de Observações"
    ) +
    theme_minimal() +
    scale_x_continuous(limits = c(0, 1)) +
    scale_y_continuous(limits = c(0, 1))
}

# Gerar plot (opcional - descomente se quiser ver o gráfico)
plot_calibration_curve(calib_results)



#9. RESUMO FINAL----

cat("\n=== RESUMO FINAL DO MODELO ===\n")
cat("Desempenho:\n")
cat("- AUC:", round(xg_metrics$auc_roc, 4), "\n")
cat("- Log Loss:", round(xg_metrics$log_loss, 4), "\n")
cat("- Brier Score:", round(xg_metrics$brier_score, 4), "\n")

cat("\nCalibração:\n")
cat("- Diferença Gols Reais vs Esperados:", round(diferenca_percentual, 2), "%\n")
cat("- Expected Calibration Error:", round(calibration_stats$ECE, 4), "\n")
cat("- Bias Médio:", round(calibration_stats$bias, 4), "\n")

cat("\nTreinamento:\n")
cat("- Épocas treinadas:", length(history_final$metrics$loss), "\n")
cat("- Melhor AUC validação:", round(max(history_final$metrics$val_auc), 4), "\n")
