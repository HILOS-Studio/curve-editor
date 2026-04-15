# hilos-runpod-semantic-edit

RunPod serverless handler for semantic image editing operations. Provides curve-driven image warping, Bezier curve resampling, and contour extraction.

## Overview

This service exposes three operations via a stateless RunPod serverless endpoint:

- **warp**: Curve-driven image warping using Gaussian-weighted grid displacement
- **resample**: Bezier curve resampling with least-squares fitting
- **contour**: Contour extraction from mask images

## Deployment

### Build the Docker image

```bash
docker build -t hilos-runpod-semantic-edit .
```

### Test locally with RunPod

```bash
# Create a test input file
cat > test_input.json << 'EOF'
{
  "input": {
    "operation": "resample",
    "control_points": [[0, 0], [10, 20], [20, 10], [30, 30], [40, 20], [50, 10], [60, 0]],
    "new_segments": 2
  }
}
EOF

# Run locally
python handler.py --test_input test_input.json
```

### Deploy to RunPod

1. Push the Docker image to a registry (Docker Hub, GitHub Container Registry, etc.)
2. Create a new Serverless Endpoint on RunPod
3. Configure the endpoint with your Docker image
4. Note the endpoint ID for use in your application

## API Reference

### Warp Operation

Performs curve-driven image warping.

**Input:**
```json
{
  "operation": "warp",
  "image_base64": "<base64-encoded PNG/JPEG>",
  "src_curves": {
    "collar": {"control_points": [[x, y], ...]},
    "upper_heel": {"control_points": [[x, y], ...]},
    ...
  },
  "dst_curves": {
    "collar": {"control_points": [[x, y], ...]},
    ...
  },
  "grid_n": 55,
  "sigma": 30
}
```

**Output:**
```json
{
  "warped_image_base64": "<base64-encoded PNG>",
  "grid_visualization": {
    "grid_xs": [...],
    "grid_ys": [...],
    "warped_gx": [[...]],
    "warped_gy": [[...]],
    "grid_n": 55
  },
  "moved_points": [[x, y], ...],
  "anchor_points": [[x, y], ...]
}
```

**Parameters:**
- `grid_n`: Grid resolution (5-200, default 55)
- `sigma`: Gaussian kernel sigma in workspace pixels (5-500, default 30)

### Resample Operation

Resamples a Bezier curve with a new segment count.

**Input:**
```json
{
  "operation": "resample",
  "control_points": [[x, y], ...],
  "new_segments": 3
}
```

**Output:**
```json
{
  "control_points": [[x, y], ...]
}
```

**Parameters:**
- `new_segments`: Number of cubic segments (1-30)

### Contour Operation

Extracts contour from a last mask image.

**Input:**
```json
{
  "operation": "contour",
  "last_image_base64": "<base64-encoded PNG>",
  "fe_image_base64": "<optional base64-encoded feather edge image>"
}
```

**Output:**
```json
{
  "contour": [[x, y], ...],
  "feather_edge": [[x, y], ...]
}
```

## Curve Names

The warp operation recognizes these curve names (from footwear semantic editing):

- `collar`
- `upper_heel`
- `upper_tongue`
- `bite`
- `ground`
- `heel`
- `toe`

## Architecture

The service is stateless - each request includes all necessary data. The client is responsible for maintaining state (current warped image, curve positions) between requests.

```
Client                          RunPod
  |                               |
  |-- POST /run {warp request} -->|
  |                               |-- decode image
  |                               |-- compute warp
  |                               |-- encode result
  |<-- {warped_image_base64} -----|
  |                               |
  |-- (client stores result) -----|
  |                               |
  |-- POST /run {next warp} ----->|
  |   (includes previous result)  |
```

## Development

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run tests

```bash
python -m pytest tests/
```

## License

Proprietary - HILOS Studio
