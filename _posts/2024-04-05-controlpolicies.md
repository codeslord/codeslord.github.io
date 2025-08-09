---
layout: post
title: Linear and Non-linear control
date: 2024-04-05 11:00
summary: Linear and Non Linear control policies.
categories: General
---

<img src="https://i.ibb.co/wQBqJnK/robot-arm.jpg" alt="robot-arm" border="0">

### Linear and Non-Linear Control Policies

#### Linear Control Policy
A linear control policy uses linear functions to map the state of a system to control actions. In this context, a linear relationship between input (state) and output (control action) can be represented as:

$$
u = Kx 
$$

where $$ u $$ is the control input, $$ K $$ is the control gain matrix, and $$ x $$ is the state vector. Linear control policies are relatively simple and can be very effective for systems that can be approximated well by linear models.

#### Non-Linear Control Policy
A non-linear control policy uses non-linear functions to map the state to control actions. Non-linear control is necessary for systems where the relationship between input and output is inherently non-linear. The non-linear control policy can be represented as:

$$
u = f(x) 
$$

where $$ f $$ is a non-linear function. Non-linear control policies are more complex and can handle a wider range of system behaviors.

### Ridge Regression
Ridge regression is a type of linear regression that includes a regularization term to prevent overfitting. The objective function for ridge regression is:

$$
\min_{w} \| Xw - y \|^2_2 + \lambda \| w \|^2_2 
$$

where $$ w $$ are the weights, $$ X $$ is the input data, $$ y $$ is the output data, and $$ \lambda $$ is the regularization parameter. The regularization term $$ \lambda \| w \|^2_2 $$ penalizes large weights to prevent overfitting.

### Ridge Regression as a Control Policy
Ridge regression can be used to derive a control policy by learning the mapping from states to control inputs. When applied to control, ridge regression helps in determining the optimal control actions that achieve desired outcomes while avoiding overfitting to specific data points.

### Two-Link Robot Arm Manipulator

In the context of a two-link robot arm manipulator, the control policies (linear or non-linear) are used to determine the torques (control actions) applied at the joints to achieve a desired trajectory or end-effector position.

#### Linear Control Policy for Two-Link Robot Arm
For a linear control policy, you might linearize the dynamics around a nominal trajectory or operating point and use a linear controller like Proportional-Derivative (PD) control. The control law might look like:

$$ 
\tau = K_p (q_d - q) + K_d (\dot{q}_d - \dot{q}) 
$$

where $$ \tau $$ is the control torque, $$ q $$ is the current joint angles, $$ q_d $$ is the desired joint angles, $$ K_p $$ and $$ K_d $$ are the proportional and derivative gain matrices.

#### Non-Linear Control Policy for Two-Link Robot Arm
For a non-linear control policy, you consider the full non-linear dynamics of the manipulator. A typical non-linear control method is computed torque control, which can be represented as:

$$ 
\tau = M(q) \ddot{q}_d + C(q, \dot{q}) \dot{q}_d + G(q) 
$$

where $$ M(q) $$ is the mass matrix, $$ C(q, \dot{q}) $$ represents Coriolis and centrifugal forces, and $$ G(q) $$ represents gravitational forces. The desired joint accelerations $$ \ddot{q}_d $$ are computed based on the desired trajectory.

### Ridge Regression in Two-Link Robot Arm Control
In the context of a two-link robot arm, ridge regression could be used to learn a linear approximation of the non-linear control policy. For example, you could collect data on the state $$ x = [q, \dot{q}] $$ and the required control torques $$ \tau $$. Using ridge regression, you can fit a model:

$$ 
\tau = Kx 
$$

where $$ K $$ is learned using ridge regression. This approach provides a linear approximation of the non-linear control policy and includes regularization to prevent overfitting.

While ridge regression is typically used in a learning context rather than a direct control policy, the model learned through ridge regression can inform a control policy by providing an approximation of the relationship between states and control actions.

### Examples of Linear and Non-Linear Control Policies

#### Linear Control Policies

1. **Proportional-Derivative (PD) Control**
   - **Application**: Position control of a motor
   - **Control Law**:
     $$
     u = K_p (r - y) + K_d \left( \frac{d}{dt} (r - y) \right)
     $$
     where $$ u $$ is the control input (voltage to the motor), $$ r $$ is the reference position, $$ y $$ is the current position, $$ K_p $$ is the proportional gain, and $$ K_d $$ is the derivative gain.

2. **Linear Quadratic Regulator (LQR)**
   - **Application**: Stabilizing an inverted pendulum
   - **Control Law**:
     $$
     u = -Kx
     $$
     where $$ u $$ is the control input (force applied), $$ K $$ is the feedback gain matrix, and $$ x $$ is the state vector (including angle and angular velocity).

3. **State-Space Control**
   - **Application**: Control of a multi-input multi-output (MIMO) system like a robotic arm
   - **Control Law**:
     $$
     u = -Kx + Kr
     $$
     where $$ u $$ is the control input, $$ K $$ is the state feedback gain matrix, $$ x $$ is the state vector, and $$ r $$ is the reference input.

#### Non-Linear Control Policies

1. **Computed Torque Control (CTC)**
   - **Application**: Control of a robotic manipulator
   - **Control Law**:
     $$
     \tau = M(q) \ddot{q}_d + C(q, \dot{q}) \dot{q}_d + G(q)
     $$
     where $$ \tau $$ is the control torque, $$ M(q) $$ is the mass matrix, $$ C(q, \dot{q}) $$ is the Coriolis and centrifugal matrix, $$ G(q) $$ is the gravity vector, $$ q $$ is the joint position, and $$ \ddot{q}_d $$ is the desired joint acceleration.

2. **Sliding Mode Control (SMC)**
   - **Application**: Robust control of a system with uncertainties, such as an underwater vehicle
   - **Control Law**:
     $$
     u = -K \cdot \text{sign}(s)
     $$
     where $$ u $$ is the control input, $$ K $$ is a constant, and $$ s $$ is the sliding surface defined as $$ s = c e + \dot{e} $$ (with $$ e $$ being the tracking error and $$ c $$ a positive constant).

3. **Adaptive Control**
   - **Application**: Control of a system with unknown parameters, like an aircraft
   - **Control Law**:
     $$
     u = \theta^T \phi(x)
     $$
     where $$ u $$ is the control input, $$ \theta $$ is the vector of adaptive parameters, and $$ \phi(x) $$ is a vector of non-linear functions of the state $$ x $$. The parameters $$ \theta $$ are updated based on the tracking error.

4. **Feedback Linearization**
   - **Application**: Control of a non-linear system, such as a quadrotor drone
   - **Control Law**:
     $$
     u = \alpha(x) + \beta(x)v
     $$
     where $$ u $$ is the control input, $$ \alpha(x) $$ and $$ \beta(x) $$ are non-linear functions of the state $$ x $$, and $$ v $$ is the new input designed to achieve linear dynamics.

5. **Neural Network Control**
   - **Application**: Control of a two-link robot arm
   - **Control Law**:
     $$
     u = f(x)
     $$
     where $$ u $$ is the control input, $$ x $$ is the state vector (e.g., joint angles and velocities), and $$ f $$ is a non-linear function represented by a trained neural network. The neural network is trained to approximate the desired control policy based on collected data.

These examples illustrate the diversity of linear and non-linear control policies. Linear control policies, such as PD control and LQR, are simpler and easier to implement but may not be suitable for systems with significant non-linear dynamics. Non-linear control policies, including neural networks, can handle complex system behaviors but are more complex to design and implement.

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fcodeslord.github.io%2Fgeneral%2F2024%2F06%2F02%2Fcontrolpolicies%2F&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)
