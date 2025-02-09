# Load necessary libraries
library(ggplot2)

library(e1071)  # For SVR
library(class)  # For KNN
library(dplyr)  # For data manipulation

# Set a random seed for reproducibility
set.seed(123)

# 1. Data Generation
# Generate synthetic data for regression
x <- seq(-3, 3, by=0.1)
y <- 2 * x + rnorm(length(x), mean=0, sd=0.5)  
data_regression <- data.frame(x = x, y = y)

# Generate synthetic data for polynomial regression
y_poly <- 1 + 2 * x^2 + rnorm(length(x), mean=0, sd=0.5)  
data_poly <- data.frame(x = x, y = y_poly)

# Generate synthetic dataset for KNN classification
set.seed(42)
x_class <- matrix(rnorm(200), ncol=2)  # 100 samples with 2 features
y_class <- as.factor(ifelse(x_class[, 1] + x_class[, 2] > 0, "Class 1", "Class 2"))  # Class labels
data_classification <- data.frame(x1 = x_class[, 1], x2 = x_class[, 2], class = y_class)


# 2. Simple Linear Regression
linear_model <- lm(y ~ x, data=data_regression)
linear_summary <- summary(linear_model)

# Plotting the Simple Linear Regression
linear_plot <- ggplot(data_regression, aes(x=x, y=y)) +
  geom_point() +
  geom_smooth(method="lm", col="blue") +
  ggtitle("Simple Linear Regression") +
  theme_minimal()

# 3. Polynomial Regression
poly_model <- lm(y ~ poly(x, 2), data=data_poly)
poly_summary <- summary(poly_model)

# Plotting the Polynomial Regression
poly_plot <- ggplot(data_poly, aes(x=x, y=y)) +
  geom_point() +
  geom_smooth(method="lm", formula=y ~ poly(x, 2), col="red") +
  ggtitle("Polynomial Regression") +
  theme_minimal()

# 4. Support Vector Regression (SVR)
svr_model <- svm(y ~ x, data=data_regression)
pred_svr <- predict(svr_model, newdata=data_regression)

# Plotting the SVR
svr_plot <- ggplot(data_regression, aes(x=x, y=y)) +
  geom_point() +
  geom_line(aes(y=pred_svr), col="green") +
  ggtitle("Support Vector Regression") +
  theme_minimal()

# 5. K-Nearest Neighbors (KNN) Classification
# Splitting the data into training and testing sets
set.seed(123)
train_indices <- sample(1:nrow(data_classification), nrow(data_classification) * 0.7)
train_data <- data_classification[train_indices, ]
test_data <- data_classification[-train_indices, ]

# KNN Model
k <- 5  
pred_knn <- knn(train = train_data[, 1:2], test = test_data[, 1:2], cl = train_data$class, k = k)

# Create a confusion matrix
confusion_matrix <- table(pred_knn, test_data$class)

# Visualizing the KNN Classification
knn_plot <- ggplot(data_classification, aes(x=x1, y=x2, color=class)) +
  geom_point(alpha=0.5) +
  ggtitle("K-Nearest Neighbors Classification") +
  scale_color_manual(values = c("Class 1" = "blue", "Class 2" = "orange")) +
  theme_minimal()

# Output results
print("Simple Linear Regression Summary:")
print(linear_summary)

print("Polynomial Regression Summary:")
print(poly_summary)

print("Confusion Matrix for KNN:")
print(confusion_matrix)

# Display all plots
print(linear_plot)
print(poly_plot)
print(svr_plot)
print(knn_plot)

