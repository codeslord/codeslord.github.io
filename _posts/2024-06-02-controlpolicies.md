---
layout: post
title: Machine Teaching - Part 1
date: 2024-06-02 11:00
summary: Linear and Non Linear control policies.
categories: General
---

<img src="https://i.ibb.co/wQBqJnK/robot-arm.jpg" alt="robot-arm" border="0">

### Linear and Non-Linear Control Policies

#### Linear Control Policy
A linear control policy uses linear functions to map the state of a system to control actions. In this context, a linear relationship between input (state) and output (control action) can be represented as:
\[ u = Kx \]
where \( u \) is the control input, \( K \) is the control gain matrix, and \( x \) is the state vector. Linear control policies are relatively simple and can be very effective for systems that can be approximated well by linear models.

#### Non-Linear Control Policy
A non-linear control policy uses non-linear functions to map the state to control actions. Non-linear control is necessary for systems where the relationship between input and output is inherently non-linear. The non-linear control policy can be represented as:
\[ u = f(x) \]
where \( f \) is a non-linear function. Non-linear control policies are more complex and can handle a wider range of system behaviors.

### Ridge Regression
Ridge regression is a type of linear regression that includes a regularization term to prevent overfitting. The objective function for ridge regression is:
\[ \min_{w} \| Xw - y \|^2_2 + \lambda \| w \|^2_2 \]
where \( w \) are the weights, \( X \) is the input data, \( y \) is the output data, and \( \lambda \) is the regularization parameter. The regularization term \( \lambda \| w \|^2_2 \) penalizes large weights to prevent overfitting.

### Ridge Regression as a Control Policy
Ridge regression can be used to derive a control policy by learning the mapping from states to control inputs. When applied to control, ridge regression helps in determining the optimal control actions that achieve desired outcomes while avoiding overfitting to specific data points.

### Two-Link Robot Arm Manipulator

In the context of a two-link robot arm manipulator, the control policies (linear or non-linear) are used to determine the torques (control actions) applied at the joints to achieve a desired trajectory or end-effector position.

#### Linear Control Policy for Two-Link Robot Arm
For a linear control policy, you might linearize the dynamics around a nominal trajectory or operating point and use a linear controller like Proportional-Derivative (PD) control. The control law might look like:
\[ \tau = K_p (q_d - q) + K_d (\dot{q}_d - \dot{q}) \]
where \( \tau \) is the control torque, \( q \) is the current joint angles, \( q_d \) is the desired joint angles, \( K_p \) and \( K_d \) are the proportional and derivative gain matrices.

#### Non-Linear Control Policy for Two-Link Robot Arm
For a non-linear control policy, you consider the full non-linear dynamics of the manipulator. A typical non-linear control method is computed torque control, which can be represented as:
\[ \tau = M(q) \ddot{q}_d + C(q, \dot{q}) \dot{q}_d + G(q) \]
where \( M(q) \) is the mass matrix, \( C(q, \dot{q}) \) represents Coriolis and centrifugal forces, and \( G(q) \) represents gravitational forces. The desired joint accelerations \( \ddot{q}_d \) are computed based on the desired trajectory.

### Ridge Regression in Two-Link Robot Arm Control
In the context of a two-link robot arm, ridge regression could be used to learn a linear approximation of the non-linear control policy. For example, you could collect data on the state \( x = [q, \dot{q}] \) and the required control torques \( \tau \). Using ridge regression, you can fit a model:
\[ \tau = Kx \]
where \( K \) is learned using ridge regression. This approach provides a linear approximation of the non-linear control policy and includes regularization to prevent overfitting.

While ridge regression is typically used in a learning context rather than a direct control policy, the model learned through ridge regression can inform a control policy by providing an approximation of the relationship between states and control actions.