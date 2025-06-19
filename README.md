# Actor & Critic Network Interface Documentation

---

## 1. Actor Network

### Observation Vector (`obs`)

The actor network input is a **45-dimensional** vector consisting of:

| Feature                  | Size | Description                                                                 |
|--------------------------|------|-----------------------------------------------------------------------------|
| Body angular velocity    | 3    | Angular velocity in the robot's body frame `[x, y, z]`                     |
| Joystick commands        | 3    | Desired base velocity `[forward (x), lateral (y), yaw rate (z)]`           |
| Gravity vector (body)    | 3    | Gravity vector `[0, 0, -1]` rotated into the robot's body frame            |
| Joint positions          | 12   | Current joint angles (network order)                                       |
| Joint velocities         | 12   | Angular velocities of joints (network order)                               |
| Previous actions         | 12   | Last action vector from the policy (network order)                         |

The network outputs a 12-dimensional action vector representing desired joint position offsets (network order)

---

## 2. Critic Network (Value Function)

### Observation Vector (`obs`)

The critic network input is a **30-dimensional** vector:

| Feature                  | Size | Description                                                               |
|--------------------------|------|---------------------------------------------------------------------------|
| Gravity vector (body)    | 3    | Gravity vector in the robot's body frame                                 |
| Body angular velocity    | 3    | Angular velocity of the robot's body frame                               |
| Joint positions          | 12   | Joint angles (network order)                                             |
| Joint velocities         | 12   | Joint angular velocities (network order)                                 |

## 2. Leg Order Conventions

### MuJoCo Default Leg Order

MuJoCo uses the following joint ordering:

| Leg Code | Leg Name     | Index Range | Joints         |
|----------|--------------|-------------|----------------|
| FR       | Front Right  | 0–2         | HAA, HFE, KFE  |
| FL       | Front Left   | 3–5         | HAA, HFE, KFE  |
| RR       | Rear Right   | 6–8         | HAA, HFE, KFE  |
| RL       | Rear Left    | 9–11        | HAA, HFE, KFE  |

### Network Expected Leg Order

The policy and value networks expect joints in this order:

| Leg Code | Leg Name     | Index Range | Joints         |
|----------|--------------|-------------|----------------|
| FL       | Front Left   | 0–2         | HAA, HFE, KFE  |
| FR       | Front Right  | 3–5         | HAA, HFE, KFE  |
| RL       | Rear Left    | 6–8         | HAA, HFE, KFE  |
| RR       | Rear Right   | 9–11        | HAA, HFE, KFE  |

