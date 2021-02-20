# Linear Regression Tutorial

## Source:
Youtube: Linear Regression in Python - Machine Learning From Scratch 02 -Python
Tutorial
by: Python Engineer
https://youtu.be/4swNt7PiamQ

## Sumamry:

## Notes:
* We approximate using the formula : y_pred = w\*x + b
* We use the __mean squared error__ as the cost function
(square the difference between the actual value and the predicted value, sum
them and divide by the no. of samples)
* We calculate the __gradient__ of the cost function with respect to w and with
  respect to b. 
* With the calculated gradient we use the __Gradient Descent__ technique = iterative method to get to the minimum
* With each iteration we have an update with the new weights and the new bias:
   w = w - alpha \* dw
   b = b - alpha \* db

## Files:
### tutorial-linear-regression.py
Generates a regression problem and plots the data

### linear-regression-tests.py
Driver program that generates and solves a randomly generated regression problem using the
mean squared error function.
