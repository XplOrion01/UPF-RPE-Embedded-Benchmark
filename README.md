# UPF-RPE-Embedded-Benchmark

Preliminary computational and resource feasibility study of a trimmed UPF-RPE
(relative pose estimation) workload for embedded platforms.

This repository contains standalone reference and benchmark implementations
derived from the study of the original UPF-RPE research codebase.

---

## 1. Purpose

The objective of this work was to investigate whether the computational
workload associated with the UPF-RPE estimator could be implemented on
resource-constrained embedded platforms.

The work was carried out as a preliminary feasibility study before integrating
the estimator with real sensors, communication interfaces, and robotic
hardware.

The benchmark was evaluated on:

- Windows PC using C++
- Windows PC using Python
- ESP32-S3
- Teensy 4.1

The main measurements were:

- execution time
- real-time computational load
- memory usage
- heap availability
- deadline misses
- numerical stability

---

## 2. Relation to the Original UPF-RPE Repository

This work was developed by studying the original UPF-RPE repository:

https://github.com/y2d2/upf_rpe

The original repository contains considerably more functionality, including
simulation, measurement generation, ROS-related components, logging,
visualization, and the complete research implementation.

For the embedded feasibility study, only the computationally relevant
estimator workload was extracted and implemented in standalone benchmark
programs.

### Important clarification

**This repository is NOT a faithful reproduction of the original UPF-RPE
implementation.**

The current implementations are trimmed, standalone reference/benchmark
implementations created for computational and resource evaluation.

They should therefore not be interpreted as a line-by-line reproduction of
the original research code.

In particular:

- the PC implementations are standalone reference implementations;
- the embedded implementations are computational benchmark versions;
- ROS, logging, visualization, and hardware I/O are not included;
- real IMU, UWB, odometry, timestamps, and communication interfaces are not
  yet integrated;
- the current implementation has not yet been validated for numerical
  equivalence against the complete original repository.

The purpose of this repository is therefore **feasibility assessment and
benchmarking**, not replacement of the original UPF-RPE implementation.

---

## 3. Benchmark Workload

The benchmark workload consists of:

- 5 independent Algebraic 4-DoF estimator instances
- 5 independent UKF tracking workload slots
- 1 UPF estimator
- 32 initial UPF particles

The workload was selected to represent the intended computational scale of
running multiple estimator instances concurrently.

The embedded implementations execute the estimator workload as a real-time
benchmark at different target frequencies.

Test frequencies include:

- 10 Hz
- 20 Hz
- 30 Hz
- 50 Hz
- 100 Hz

---


