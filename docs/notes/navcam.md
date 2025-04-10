# NavCam

A camera on Astrobee to capture images of the environment.

---

## Parameters

- 130° FOV
- 1280x960 resolution
- 4:3 aspect ratio (1280x960)
- Fixed focus
- 1.2M pixel
- RGB

---

## Cover Range

When the camera is positioned perpendicular to the plane, the range a camera can cover depends on:

- **FOV Angle**: The angular extent of the camera's view, typically given as horizontal (HFOV), vertical (VFOV), or diagonal (DFOV).
- **Distance**: The distance from the camera to the target plane (in meters).
- **Sensor Aspect Ratio**: Influences the relative sizes of HFOV and VFOV (4:3).

The horizontal and vertical cover ranges can be calculated respectively as:

---

## Step to calculate

1. Estimate HFOV and VFOV, which depend on the sensor's aspect ratio.

$$
\text{HFOV} = 130° \times \frac{4/3}{\sqrt{1 + (4/3)^2}} = 104° \\
\text{VFOV} = 130° \times \frac{3/4}{\sqrt{1 + (3/4)^2}} = 78°
$$

1. Calculate Field of View Range.

$$
\text{Horizontal Range} = 2 \times \tan\left(\frac{\text{HFOV}}{2}\right) \times \text{Distance}
$$

$$
\text{Vertical Range} = 2 \times \tan\left(\frac{\text{VFOV}}{2}\right) \times \text{Distance}
$$

3. Results

- Distance: 0.7m
  - Range: 1.79m x 1.13m
- Distance: 0.6m
  - Range: 1.54m x 0.97m
- Distance: 0.5m
  - Range: 1.28m x 0.81m

---

## Comparison with Kibo-RPC Area

- Area 1: 1.06m x 0.75m
- Area 2: 1.25m x 0.75m
- Area 3: 1.25m x 0.95m
- Area 4: 1.25m x 0.975m

## Reference

- [Multi-Agent 3D Map Reconstruction and Change Detection in Microgravity with Free-Flying Robots](https://arxiv.org/html/2311.02558v2)
