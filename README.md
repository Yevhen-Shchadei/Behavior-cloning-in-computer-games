# Snake & Flappy Bird — Behavior Cloning Dataset Package

This project contains two small games prepared for collecting player demonstrations and training neural networks with **Behavior Cloning**:

- **Snake** — records player actions and trains several neural models to imitate the player.
- **Flappy Bird** — records gameplay states/actions and trains a neural model for flap/no-flap prediction.

The package is intended for experiments with supervised imitation learning from human gameplay data.

---

## Project structure

```text
.
├── requirements.txt
├── snake_game/
│   ├── game.py
│   ├── train.py
│   ├── config.py
│   ├── ai.py
│   ├── yarik_topology.py
│   ├── snake_dataset.csv
│   ├── snake_smallnet.pth
│   ├── snake_biggernet.pth
│   └── snake_res_model.pth
│
└── flappy_game/
    ├── game.py
    ├── train_model.py
    ├── train_model_Y.py
    ├── ai_agent.py
    ├── recorder.py
    ├── config.py
    └── data/
        ├── flappy_dataset.csv
        ├── flappy_model.pt
        └── flappy_model_meta.json
```

---

## Installation

Open a terminal in the project root folder and install dependencies:

```bash
python -m pip install -r requirements.txt
```

If you want to install dependencies manually:

```bash
python -m pip install pygame numpy pandas torch scikit-learn
```

Recommended Python version: **Python 3.10+**.

---

# Snake Behavior Cloning

## Run Snake

From the project root folder:

```bash
python -m snake_game
```

Alternative direct launch:

```bash
python snake_game/game.py
```

## Controls

| Key | Action |
|---|---|
| `W` / `Arrow Up` | Move up |
| `S` / `Arrow Down` | Move down |
| `A` / `Arrow Left` | Move left |
| `D` / `Arrow Right` | Move right |
| `Enter` / `Space` | Start or restart game |
| `Esc` | Quit / return |
| `Ctrl + Z` | Undo last recorded row |

Dataset recording is enabled by default in `snake_game/game.py`:

```python
RECORD_DATASET = True
```

## Snake dataset

The dataset is saved to:

```text
snake_game/snake_dataset.csv
```

Each row represents one game step and contains the current state features plus the action performed by the player.

### Action classes

| Value | Meaning |
|---:|---|
| `0` | Go straight |
| `1` | Turn left |
| `2` | Turn right |

### Recorded features

| Column | Description |
|---|---|
| `dir_up` | Snake is currently moving up |
| `dir_down` | Snake is currently moving down |
| `dir_left` | Snake is currently moving left |
| `dir_right` | Snake is currently moving right |
| `danger_straight` | Collision danger if the snake goes straight |
| `danger_left` | Collision danger if the snake turns left |
| `danger_right` | Collision danger if the snake turns right |
| `dx_food` | Relative horizontal direction/distance to food |
| `dy_food` | Relative vertical direction/distance to food |
| `action` | Target action: `0`, `1`, or `2` |

Current training code uses **9 input features** and one target column.

## Train Snake models

After collecting enough data, run:

```bash
python snake_game/train.py
```

The training script trains and saves several models:

| Model | Architecture | Output file |
|---|---|---|
| `SmallNet` | `9 -> 16 -> 3` | `snake_game/snake_smallnet.pth` |
| `BiggerNet` | `9 -> 32 -> 16 -> 3` | `snake_game/snake_biggernet.pth` |
| `ResidualNet` | residual network, hidden size 64 | `snake_game/snake_res_model.pth` |

Recommended dataset size:

| Dataset size | Expected quality |
|---:|---|
| `< 1000` samples | Too small, unstable model |
| `3000+` samples | Minimum usable dataset |
| `5000–10000+` samples | Better imitation quality |

---

# Flappy Bird Recording

## Run Flappy Bird

From the project root folder:

```bash
python -m flappy_game
```

Alternative direct launch:

```bash
python flappy_game/game.py
```

## Controls

| Key | Action |
|---|---|
| `Space` / `Arrow Up` | Flap |
| `R` | Start/stop recording |
| `S` | Save current recording buffer |
| `Z` | Undo last recorded row |
| `A` | Toggle AI mode |
| `Esc` | Quit |

Recording is disabled by default. Press `R` in-game to start recording.

The default recording setting is stored in `flappy_game/config.py`:

```python
RECORD_DEFAULT = False
```

## How to record Flappy dataset

1. Run the game:

   ```bash
   python -m flappy_game
   ```

2. Press `R` to start recording.
3. Play the game using `Space` or `Arrow Up`.
4. Press `R` again to stop recording and save the data.
5. You can also press `S` to manually flush the current recording buffer.

## Flappy dataset

The dataset is saved to:

```text
flappy_game/data/flappy_dataset.csv
```

The CSV contains one row per game step. It is prepared for direct use with pandas, scikit-learn, or PyTorch.

### CSV columns

| Column | Description |
|---|---|
| `bird_x_norm` | Normalized bird x-coordinate: `bird_x / SCREEN_W` |
| `bird_y_norm` | Normalized bird y-coordinate: `bird_y / SCREEN_H` |
| `bird_vel_norm` | Normalized vertical velocity, clipped to `[-1, 1]` |
| `dist_to_pipe_x_norm` | Normalized horizontal distance to the next pipe |
| `pipe_top_y_norm` | Normalized top boundary of the pipe gap |
| `pipe_bottom_y_norm` | Normalized bottom boundary of the pipe gap |
| `pipe_gap_center_y_norm` | Normalized y-coordinate of the pipe gap center |
| `action` | Target action: `0 = no flap`, `1 = flap` |

### Original state format

Internally, the game state has 7 values:

```python
[
    bird_x,
    bird_y,
    bird_vel,
    dist_to_pipe_x,
    pipe_top_y,
    pipe_bottom_y,
    pipe_gap_center_y,
]
```

The recorder saves normalized values to CSV.

## Train Flappy model

After recording data, run:

```bash
python flappy_game/train_model.py
```

The script reads:

```text
flappy_game/data/flappy_dataset.csv
```

and saves:

```text
flappy_game/data/flappy_model.pt
flappy_game/data/flappy_model_meta.json
```

The training script derives 9 model features from the recorded CSV:

| Feature | Meaning |
|---|---|
| `bird_center_y` | Bird vertical position |
| `bird_vel` | Bird vertical velocity |
| `dist_x` | Distance to next pipe |
| `pipe_top_y` | Top gap boundary |
| `pipe_bottom_y` | Bottom gap boundary |
| `pipe_gap_center` | Center of the pipe gap |
| `bird_to_gap_center` | Difference between bird y-position and gap center |
| `bird_to_top` | Difference between bird y-position and top boundary |
| `bird_to_bottom` | Difference between bottom boundary and bird y-position |

Target classes:

| Value | Meaning |
|---:|---|
| `0` | No flap |
| `1` | Flap |

Recommended dataset size:

| Dataset size | Expected quality |
|---:|---|
| `< 1000` samples | Too small for reliable behavior |
| `3000+` samples | Minimum usable dataset |
| `5000–10000+` samples | Better imitation quality |

---

## Quick start

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Record Snake gameplay:

```bash
python -m snake_game
```

Train Snake models:

```bash
python snake_game/train.py
```

Record Flappy Bird gameplay:

```bash
python -m flappy_game
```

Train Flappy Bird model:

```bash
python flappy_game/train_model.py
```

---

## Notes

- Both projects use supervised learning from recorded player actions.
- The quality of the trained model depends heavily on dataset size and consistency.
- Bad or random gameplay produces bad imitation behavior.
- Try to record clean gameplay where the player reacts correctly and avoids unnecessary moves.
- If the model behaves poorly, collect more data and retrain.

---

## Output files summary

| Project | File | Description |
|---|---|---|
| Snake | `snake_game/snake_dataset.csv` | Recorded Snake dataset |
| Snake | `snake_game/snake_smallnet.pth` | Trained SmallNet weights |
| Snake | `snake_game/snake_biggernet.pth` | Trained BiggerNet weights |
| Snake | `snake_game/snake_res_model.pth` | Trained ResidualNet weights |
| Flappy Bird | `flappy_game/data/flappy_dataset.csv` | Recorded Flappy dataset |
| Flappy Bird | `flappy_game/data/flappy_model.pt` | Trained Flappy model |
| Flappy Bird | `flappy_game/data/flappy_model_meta.json` | Metadata for trained Flappy model |
