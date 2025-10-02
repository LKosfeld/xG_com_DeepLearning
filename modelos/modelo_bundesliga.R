#Carregamento dos dados----
library(tidyverse)
library(keras)
bundesliga_preparados <- read.csv("dados/preparados/bundesliga_preparados.csv")

#1. PREPARAÇÃO DOS DADOS----

features_df <- bundesliga_preparados %>% select(-is_goal)
labels_vector <- bundesliga_preparados$is_goal

##Transforma features categóricas em numéricas (one-hot encoding) e cria a matriz
features_matrix <- model.matrix(~ . -1, data = features_df)

##Divisão em conjuntos de treino (80%) e teste (20%)
set.seed(18)
train_indices <- sample(1:nrow(features_matrix), size = 0.8 * nrow(features_matrix))

train_data <- features_matrix[train_indices, ]
train_labels <- labels_vector[train_indices]
test_data <- features_matrix[-train_indices, ]
test_labels <- labels_vector[-train_indices]

##Cálculo dos pesos para classes
neg <- sum(train_labels == 0)
pos <- sum(train_labels == 1)
total <- neg + pos

# Fórmula para balancear os pesos
weight_for_0 <- (1 / neg) * (total / 2.0)
weight_for_1 <- (1 / pos) * (total / 2.0)
class_weights <- list("0" = weight_for_0, "1" = weight_for_1)

cat("Pesos calculados -> Classe 0 (Não Gol):", round(weight_for_0, 2), 
    "| Classe 1 (Gol):", round(weight_for_1, 2), "\n\n")


#2. CONSTRUÇÃO DO MODELO----
build_model <- function() {
  model <- keras_model_sequential() %>%
    ##Camada de entrada com regularização L2
    layer_dense(units = 64, activation = "relu",
                kernel_regularizer = regularizer_l2(0.01), 
                input_shape = c(ncol(train_data))) %>%
    ##Camada de Dropout
    layer_dropout(rate = 0.2) %>% 
    
    ##Camada oculta com regularização L2
    layer_dense(units = 32, activation = "relu",
                kernel_regularizer = regularizer_l2(0.01)) %>% 
    ##Outra camada de Dropout
    layer_dropout(rate = 0.2) %>% 
    
    ##Camada de saída com ativação sigmoid para probabilidade
    layer_dense(units = 1, activation = "sigmoid")
  
  model %>% compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_rmsprop(),
    metrics = c("accuracy", "AUC")
  )
  
  return(model)
}


#4. VALIDAÇÃO CRUZADA K-FOLD----

k <- 4
num_epochs <- 150

indices <- sample(1:nrow(train_data))
folds <- cut(indices, breaks = k, labels = FALSE)

all_val_auc_histories <- list()
all_epochs_ran <- c()


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
  
  ##Callback para parar o treino se a perda na validação não melhorar
  early_stop <- callback_early_stopping(monitor = "val_loss", patience = 15)
  
  history <- model %>% fit(
    partial_train_data, partial_train_labels,
    validation_data = list(val_data, val_labels),
    epochs = num_epochs, 
    batch_size = 16, 
    verbose = 0,
    callbacks = list(early_stop),
    class_weight = class_weights
  )
  
  all_val_auc_histories[[i]] <- history$metrics$val_auc
  epochs_ran <- length(history$metrics$val_loss) # Número de épocas que realmente rodaram
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

##Treino final
history_final <- model_final %>% fit(
  train_data_scaled, train_labels,
  epochs = ideal_epochs,
  batch_size = 16, 
  verbose = 1,
  class_weight = class_weights
)


#5. AVALIAÇÃO FINAL NO CONJUNTO DE TESTE----
results <- model_final %>% evaluate(test_data_scaled, test_labels)
print(results)


#6. Avaliação de Features----
padrao_nomes <- "^(play_pattern\\.name|position\\.name|shot\\.type\\.name|shot\\.technique\\.name|shot\\.body_part\\.name)"

aggregated_importance_df <- feature_importance_df %>%
  ##Cria uma nova coluna 'feature_original' extraindo o prefixo de cada nome.
  mutate(
    feature_original = str_extract(feature, padrao_nomes)
  ) %>%
  mutate(
    feature_original = ifelse(is.na(feature_original), feature, feature_original)
  ) %>%
  ##Agrupar1 pela feature original e somar as importâncias.
  group_by(feature_original) %>%
  summarise(total_importance = sum(importance)) %>%
  ungroup()

aggregated_importance_df %>%
  arrange(desc(total_importance)) %>%
  mutate(feature_original = reorder(feature_original, total_importance)) %>%
  ggplot(aes(x = feature_original, y = total_importance)) +
  geom_col(fill = "darkcyan") +
  coord_flip() +
  labs(
    title = "Importância Agregada das Features Originais",
    subtitle = "Soma da queda na AUC para cada grupo de features",
    x = "Feature Original",
    y = "Importância Total (Soma da Queda na AUC)"
  ) +
  theme_minimal()