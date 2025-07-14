# Optimization of Mechatronic Systems

A comprehensive optimization project focusing on mechatronic systems, including robot arm control and vehicle suspension optimization using advanced numerical optimization techniques.

## Quick Start

1. **Setup environment**:

   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Run the project**:

   ```bash
   jupyter notebook robot_arm.ipynb
   jupyter notebook semiactive_quarter_car.ipynb
   ```

## Project Files

- `robot_arm.ipynb` - Two-link planar robot arm optimization analysis
- `semiactive_quarter_car.ipynb` - Quarter-car suspension system optimization
- `Assignment1.pdf` - First assignment (suspension) report
- `Assignment2.pdf` - Second assignment (robot arm) report
- `objective_values.npy` - Stored optimization results
- `images/` - Generated plots and visualizations for quarter-car analysis
- `imgs/` - Generated plots and visualizations for robot arm analysis

## Project Overview

This project consists of two main optimization problems in mechatronic systems:

### 1. Robot Arm Optimization

Analysis of a two-link planar robot arm with the following objectives:

- **Acceleration-controlled robot**: States are joint positions and velocities, controls are desired joint accelerations
- **Torque-controlled robot**: States are joint positions and velocities, controls are desired joint torques
- **Optimal control problems**: Minimizing mean-square control effort while moving between specified positions
- **Time-optimal control**: Finding minimum-time trajectories with control bounds
- **Self-collision avoidance**: Ensuring the robot avoids self-collision during motion

### 2. Quarter-Car Suspension System

Optimization of a mass-spring-damper quarter-car suspension system:

- **Passive suspension**: Fixed damping coefficient optimization
- **Semiactive suspension**: Dynamic damping control using skyhook policy
- **Step disturbance analysis**: Response to road displacement inputs
- **Parameter optimization**: Finding optimal damping values for different scenarios

## Results

Multiple optimization scenarios were implemented and analyzed:

- **Robot Arm Control**:
  - Acceleration vs. torque control comparison
  - Time-optimal trajectory generation
  - Convexity analysis of optimization problems

- **Suspension Optimization**:
  - Passive vs. semiactive suspension performance
  - Skyhook control policy implementation
  - Multi-scenario optimization (1cm, 5cm, 10cm disturbances)

For complete methodology, analysis, and results, see the individual notebook files and assignment PDFs.

---

## Authors

- Andrea Alboni
- Cheng Cui
