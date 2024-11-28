---
layout: post
title: A Mathematically Grounded Journey Through 3D Gaussian Splatting
date: 2024-01-02 11:00
summary: The world of 3D rendering and reconstruction has undergone a transformative leap with the advent of 3D Gaussian Splatting. Introduced in the groundbreaking paper "3D Gaussian Splatting for Real-Time Radiance Field Rendering," this technique promises photorealistic rendering of scenes learned from sparse image samples, all in real-time.
categories: General
---

<img src="https://i.ibb.co/bFR1z6r/an-image-of-a-3d-rendering-using.jpg" alt="an-image-of-a-3d-rendering" border="0">


# **A Mathematically Grounded Journey Through 3D Gaussian Splatting**

## What is 3D Gaussian Splatting?
At its core, 3D Gaussian Splatting is a rasterization technique that diverges from traditional triangle-based rendering methods. Instead of rasterizing triangles, it uses gaussians as the basic unit of rendering.

Imagine a storyteller describing a lush, dynamic world—not with traditional tools like pencils or polygons, but through the lens of millions of shimmering, floating ellipsoids. These ellipsoids are the building blocks of **3D Gaussian Splatting**, a groundbreaking technique for reconstructing and rendering photorealistic 3D scenes in real time. Let’s journey together into this world, blending intuition and mathematics to unravel its secrets.

---

## **Meet the Gaussians: The Story’s Characters**

In this tale, **3D Gaussians** are our primary characters. Each Gaussian is like a lantern, emitting not just light but also information about color, position, and transparency. These parameters define their identity:
- **Position $$\mu$$:** Where the Gaussian resides in 3D space.
- **Covariance $$\Sigma$$:** Describes the shape and orientation of the Gaussian. Think of this as an ellipsoid stretched, squashed, or rotated in space.
- **Opacity $$\alpha$$:** How much light passes through this Gaussian.
- **Color $$c$$:** The hue of the lantern’s glow.

Mathematically, a Gaussian $$G(x)$$ in 3D is defined as:

$$
G(x) = \exp\left(-\frac{1}{2}(x - \mu)^T \Sigma^{-1} (x - \mu)\right),
$$

where:
- $$x$$ is a point in space,
- $$\mu$$ is the center of the Gaussian,
- $$\Sigma$$ is its covariance matrix, which determines its size and shape.

---

## **Setting the Stage: Scene Initialization**

Every story begins somewhere, and here it starts with a sparse **point cloud** created through **Structure from Motion (SfM)**. SfM, like a cartographer, derives a rough 3D map from photos. Each point in the cloud is like a faint whisper of the scene, waiting to be transformed into a Gaussian.

---

## **Act I: Transforming Points into Gaussians**

### **From Points to Lanterns**
To light up our scene, we transform each point into a Gaussian. Covariance $$\Sigma$$ starts simple, initialized as an isotropic Gaussian (equal size in all directions):

$$
\Sigma = \sigma^2 I,
$$

where $$I$$ is the identity matrix and $$\sigma$$ controls the initial scale. Over time, the ellipsoid morphs as we optimize the scene.

---

## **Act II: Rendering the Scene**

The magic of Gaussian Splatting is in how these lanterns are projected onto the viewer’s screen.

### **Projection to 2D**
To render a 3D Gaussian, it must be projected into the camera’s 2D plane. The new covariance matrix in 2D, $$\Sigma'$$, is computed using the camera’s projection matrix $$W$$:

$$
\Sigma' = J W \Sigma W^T J^T,
$$

where $$J$$ is the Jacobian of the affine transformation. This projects the 3D ellipsoid into an anisotropic 2D Gaussian—what you see on the screen.

### **Blending the Light**
When many Gaussians overlap, their contributions are blended to determine the pixel’s color. This process is described by **alpha blending**:

$$
C = \sum_{i=1}^N T_i \alpha_i c_i,
$$

where:
- $$C$$ is the final pixel color,
- $$\alpha_i$$ is the transparency of Gaussian $$i$$,
- $$c_i$$ is the color of Gaussian $$i$$,
- $$T_i$$ is the transmittance, calculated as:
  
  $$T_i = \prod_{j=1}^{i-1} (1 - \alpha_j)$$
  
This ensures that closer Gaussians contribute more to the pixel color, respecting depth.

---

## **Act III: Training the Lanterns**

Like actors rehearsing their lines, Gaussians refine their positions, shapes, and colors to create a believable scene. This is done through **optimization**.

### **Loss Function**
The optimization compares the rendered image to the ground truth image using a loss function. A typical choice combines an L1 error with structural similarity (D-SSIM):

$$
L = (1 - \lambda) L_1 + \lambda L_{D-SSIM}.
$$

### **Gradient Descent**
Each Gaussian adjusts its parameters—position $$\mu$$, covariance $$\Sigma$$, and opacity $$\alpha$$—through **Stochastic Gradient Descent (SGD)**. Covariance updates are constrained to ensure it remains a valid positive semi-definite matrix:

$$
\Sigma = R S^2 R^T,
$$

where $$R$$ is a rotation matrix and $$S$$ is a scaling matrix.

---

## **Act IV: Adaptive Scene Refinement**

The story doesn’t end with static characters. Some Gaussians split into smaller ones to capture finer details, while others merge or fade away if redundant.

### **Splitting and Cloning**
- **Splitting:** Large Gaussians in detailed regions are split into two:

  $$\Sigma_{\text{new}} = \frac{1}{2} \Sigma_{\text{old}}$$
  
- **Cloning:** Small Gaussians in under-reconstructed regions are duplicated and repositioned along the gradient direction.

### **Pruning**
Transparent Gaussians $$\alpha < \epsilon$$ are removed, reducing computational load.

---

## **Curtain Call: Real-Time Rendering**

Once trained, the optimized Gaussians are rendered in real time. Thanks to GPU acceleration and fast sorting algorithms like radix sort, even millions of Gaussians can be efficiently processed. The final output is a photorealistic 3D scene.

---

## **A Story Well Told**

Through the lens of 3D Gaussian Splatting, we see a harmonious blend of mathematics, optimization, and rendering artistry. By combining the elegance of Gaussians with GPU-driven speed, this technique redefines what’s possible in real-time 3D reconstruction and rendering. The future of graphics may indeed belong to these luminous, mathematical storytellers.
