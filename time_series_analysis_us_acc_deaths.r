# Load the datasets package (if not already loaded)
library(datasets)

# Load the USAccDeaths dataset
data(USAccDeaths)

# Plot the time series
plot(USAccDeaths, main = "US Accidental Deaths")

# Plot the ACF plot
acf(USAccDeaths, main = "ACF Plot")

# Plot the seasonal subseries plot
library(ggplot2)
ggseasonplot(USAccDeaths, year.labels = TRUE, main = "Seasonal Subseries Plot")

# Split the data into training and test sets
set.seed(123) # for reproducibility
n <- length(USAccDeaths)
train_size <- round(0.7 * n)
train <- USAccDeaths[1:train_size]
test <- USAccDeaths[(train_size+1):n]

# Estimate an ETS model on the training data
library(forecast)
model_M3 <- ets(train, model = "M3")

# Test for residuals diagnostic checking
checkresiduals(model_M3)

# Evaluate several models based on the test data set
model_M1 <- ets(train, model = "M1")
model_M2 <- ets(train, model = "M2")
model_M4 <- ets(train, model = "M4")
model_M5 <- ets(train, model = "M5")

accuracy_M1 <- accuracy(forecast(model_M1, h = length(test)))
accuracy_M2 <- accuracy(forecast(model_M2, h = length(test)))
accuracy_M3 <- accuracy(forecast(model_M3, h = length(test)))
accuracy_M4 <- accuracy(forecast(model_M4, h = length(test)))
accuracy_M5 <- accuracy(forecast(model_M5, h = length(test)))

# Find the best model
accuracy_df <- data.frame(
  model = c("M1", "M2", "M3", "M4", "M5"),
  RMSE = c(accuracy_M1[2], accuracy_M2[2], accuracy_M3[2], accuracy_M4[2], accuracy_M5[2])
)
best_model <- accuracy_df[which.min(accuracy_df$RMSE), "model"]

# Forecast 10 points ahead using the best model
best_model_fit <- ets(USAccDeaths, model = best_model)
best_model_fcst <- forecast(best_model_fit, h = 10)
best_model_fcst_trans <- invBoxCox(best_model_fcst$mean, model_M3$lambda)
best_model_fcst$mean <- best_model_fcst_trans

# Plot the forecast
plot(best_model_fcst, main = paste("Forecast using best model (", best_model, ")"))
