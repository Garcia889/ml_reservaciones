library(tidyverse)
library(xgboost)


train_data <- read.csv("/Users/ximenapaz/Documents/Ciencia de Datos/Aprendizaje de Máquina/Concurso Hoteles/hoteles-entrena.csv")
test_data <- read.csv("/Users/ximenapaz/Documents/Ciencia de Datos/Aprendizaje de Máquina/Concurso Hoteles/hoteles-prueba.csv")

head(train_data)
head(test_data)


#Fechas
train_data <- train_data %>%
  mutate(arrival_date = as.Date(arrival_date),
         arrival_month = lubridate::month(arrival_date),
         arrival_day = lubridate::day(arrival_date))

test_data <- test_data %>%
  mutate(arrival_date = as.Date(arrival_date),
         arrival_month = lubridate::month(arrival_date),
         arrival_day = lubridate::day(arrival_date))

#Categorical a factores
categorical_columns <- c("hotel", "meal", "country", "market_segment", 
                         "distribution_channel", "reserved_room_type", 
                         "assigned_room_type", "deposit_type", "agent", 
                         "company", "customer_type")

train_data[categorical_columns] <- lapply(train_data[categorical_columns], function(x) as.numeric(as.factor(x)))
test_data[categorical_columns] <- lapply(test_data[categorical_columns], function(x) as.numeric(as.factor(x)))

train_data$children <- ifelse(train_data$children == "none", 0, 1)

# Sustitución de variable 
train_data$required_car_parking_spaces <- as.numeric(replace(train_data$required_car_parking_spaces, 
                                                             train_data$required_car_parking_spaces == "none", 0))
test_data$required_car_parking_spaces <- as.numeric(replace(test_data$required_car_parking_spaces, 
                                                            test_data$required_car_parking_spaces == "none", 0))

# Eliminamos columna 
train_data <- train_data %>% select(-arrival_date)
test_data <- test_data %>% select(-arrival_date)

# Separar las variables predictoras y la variable objetivo
X_train <- train_data %>% select(-children)  
y_train <- train_data$children

# Conjunto prueba
X_test <- test_data %>% select(-id)

# 
X_train[] <- lapply(X_train, function(x) {
  if (is.character(x)) {
    as.numeric(as.factor(x))
  } else {
    as.numeric(x)
  }
})

X_test[] <- lapply(X_test, function(x) {
  if (is.character(x)) {
    as.numeric(as.factor(x))
  } else {
    as.numeric(x)
  }
})

# xgboost
dtrain <- xgb.DMatrix(data = as.matrix(X_train), label = y_train)
dtest <- xgb.DMatrix(data = as.matrix(X_test))

ratio_class <- sum(y_train == 0) / sum(y_train == 1)

params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  eta = 0.1,
  max_depth = 6,
  scale_pos_weight = ratio_class  # Balanceo de clases basado en la proporción
)

# Entrenar el modelo
modelo <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 500,
  early_stopping_rounds = 10,
  watchlist = list(train = dtrain),
  verbose = 1
)

# Predicciones
test_data$prob <- predict(modelo, dtest)

resultado <- test_data %>% select(id, prob)
write.csv(resultado, "/Users/ximenapaz/Documents/Ciencia de Datos/Aprendizaje de Máquina/Concurso Hoteles/Hoteles-predicciones", row.names = FALSE)
print("Predicciones guardadas en 'resultado_predicciones.csv'")
