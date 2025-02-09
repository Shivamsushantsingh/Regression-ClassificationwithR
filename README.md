# Regression-Classification with R
Libraries and Setup
# Load necessary libraries
library(ggplot2)
library(e1071)  # For SVR
library(class)  # For KNN
library(dplyr)  # For data manipulation
ggplot2 is used for data visualization.
e1071 provides the Support Vector Regression (SVR) algorithm.
class contains the K-Nearest Neighbors (KNN) classifier.
dplyr is used for data manipulation tasks like filtering, transforming, and summarizing data.

# Set a random seed for reproducibility
set.seed(123)
set.seed(123) ensures that the random number generation is reproducible across sessions, i.e., you'll get the same results each time you run the code with this seed.
Data Generation
1. Regression Data (for Linear and SVR models)

# Generate synthetic data for regression
x <- seq(-3, 3, by=0.1)
y <- 2 * x + rnorm(length(x), mean=0, sd=0.5)  
data_regression <- data.frame(x = x, y = y)
x generates a sequence of numbers from -3 to 3 with a step of 0.1.
y is calculated as a linear relationship to x, but with added noise (rnorm() adds random noise from a normal distribution with mean 0 and standard deviation 0.5).
data_regression combines x and y into a data frame, which will be used for regression tasks.
2. Polynomial Regression Data

# Generate synthetic data for polynomial regression
y_poly <- 1 + 2 * x^2 + rnorm(length(x), mean=0, sd=0.5)  
data_poly <- data.frame(x = x, y = y_poly)
y_poly is created using a quadratic function (polynomial of degree 2) with some added noise.
data_poly contains the data points (x, y_poly) to be used for polynomial regression.
3. Classification Data (for KNN)

# Generate synthetic dataset for KNN classification
set.seed(42)
x_class <- matrix(rnorm(200), ncol=2)  # 100 samples with 2 features
y_class <- as.factor(ifelse(x_class[, 1] + x_class[, 2] > 0, "Class 1", "Class 2"))  # Class labels
data_classification <- data.frame(x1 = x_class[, 1], x2 = x_class[, 2], class = y_class)
x_class generates a matrix of 100 rows and 2 columns filled with random numbers from a normal distribution.
y_class creates a binary classification label based on the sum of the two features (x_class[, 1] + x_class[, 2]). If the sum is positive, it gets labeled as "Class 1", otherwise "Class 2".
data_classification combines x1, x2 (the features), and class (the target) into a data frame for use in KNN classification.
Models and Plots
1. Simple Linear Regression

# Simple Linear Regression
linear_model <- lm(y ~ x, data=data_regression)
linear_summary <- summary(linear_model)
lm(y ~ x, data=data_regression) fits a linear regression model where y is predicted by x using the data_regression dataset.
summary(linear_model) provides a detailed summary of the linear model, including coefficients, R-squared value, and other statistics.

# Plotting the Simple Linear Regression
linear_plot <- ggplot(data_regression, aes(x=x, y=y)) +
  geom_point() +
  geom_smooth(method="lm", col="blue") +
  ggtitle("Simple Linear Regression") +
  theme_minimal()
ggplot() creates a plot using data_regression with x and y mapped to axes.
geom_point() adds scatter plot points to the graph.
geom_smooth(method="lm", col="blue") adds a linear regression line to the plot, in blue color.
ggtitle() sets the title for the plot.
theme_minimal() applies a minimalist theme to the plot.
2. Polynomial Regression

# Polynomial Regression
poly_model <- lm(y ~ poly(x, 2), data=data_poly)
poly_summary <- summary(poly_model)
lm(y ~ poly(x, 2), data=data_poly) fits a polynomial regression model where y is predicted by a polynomial of degree 2 for x.
summary(poly_model) outputs the summary of the polynomial regression model.

# Plotting the Polynomial Regression
poly_plot <- ggplot(data_poly, aes(x=x, y=y)) +
  geom_point() +
  geom_smooth(method="lm", formula=y ~ poly(x, 2), col="red") +
  ggtitle("Polynomial Regression") +
  theme_minimal()
This block is similar to the linear regression plot but adds a polynomial regression line in red color using geom_smooth().
3. Support Vector Regression (SVR)

# Support Vector Regression (SVR)
svr_model <- svm(y ~ x, data=data_regression)
pred_svr <- predict(svr_model, newdata=data_regression)
svm(y ~ x, data=data_regression) fits a Support Vector Regression model.
predict(svr_model, newdata=data_regression) predicts the y values using the SVR model and the input data data_regression.

# Plotting the SVR
svr_plot <- ggplot(data_regression, aes(x=x, y=y)) +
  geom_point() +
  geom_line(aes(y=pred_svr), col="green") +
  ggtitle("Support Vector Regression") +
  theme_minimal()
The geom_line() plots the predicted values (pred_svr) from the SVR model as a green line over the scatter plot of the original data.
4. K-Nearest Neighbors (KNN) Classification

# Splitting the data into training and testing sets
set.seed(123)
train_indices <- sample(1:nrow(data_classification), nrow(data_classification) * 0.7)
train_data <- data_classification[train_indices, ]
test_data <- data_classification[-train_indices, ]
sample() randomly selects 70% of the data to be used for training, leaving the rest for testing. train_data and test_data are split accordingly.

# KNN Model
k <- 5  
pred_knn <- knn(train = train_data[, 1:2], test = test_data[, 1:2], cl = train_data$class, k = k)
knn() applies the K-Nearest Neighbors algorithm. It uses the train_data features (train_data[, 1:2]), tests on the test_data features (test_data[, 1:2]), and predicts class labels using the train_data$class values. k=5 sets the number of neighbors to consider.

# Create a confusion matrix
confusion_matrix <- table(pred_knn, test_data$class)
table() creates a confusion matrix to compare the predicted values (pred_knn) with the actual class labels in test_data$class.

# Visualizing the KNN Classification
knn_plot <- ggplot(data_classification, aes(x=x1, y=x2, color=class)) +
  geom_point(alpha=0.5) +
  ggtitle("K-Nearest Neighbors Classification") +
  scale_color_manual(values = c("Class 1" = "blue", "Class 2" = "orange")) +
  theme_minimal()
geom_point(alpha=0.5) creates a scatter plot of the classification data, with the points colored by their class labels.
scale_color_manual() specifies the colors to be used for each class label ("Class 1" is blue, "Class 2" is orange).
Output Results

# Output results
print("Simple Linear Regression Summary:")
print(linear_summary)

print("Polynomial Regression Summary:")
print(poly_summary)

print("Confusion Matrix for KNN:")
print(confusion_matrix)
These lines print the summaries of the regression models and the confusion matrix for the KNN classification.

# Display all plots
print(linear_plot)
print(poly_plot)
print(svr_plot)
print(knn_plot)
Finally, the four generated plots are printed: linear regression, polynomial regression, SVR, and KNN classification.
Summary
The code demonstrates the implementation and visualization of different models: simple linear regression, polynomial regression, support vector regression (SVR), and K-Nearest Neighbors (KNN) classification using synthetic data. It involves:

Generating data for regression and classification.
Fitting models to the data.
Visualizing the model results.
Evaluating the KNN classification performance with a confusion matrix.
