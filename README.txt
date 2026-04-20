Snake Behavior Cloning — Ready Package

Files:
- snake.py              -> Snake game with dataset recording
- train_snake_bc.py     -> Train 2 neural models on recorded dataset

How to install:
1. Open terminal in this folder
2. Install dependencies:
   python -m pip install pygame pandas torch

How to collect dataset:
1. Run:
   python snake.py
2. Play several games
3. Dataset will be saved automatically into:
   snake_dataset.csv

Recorded action classes:
- 0 = straight
- 1 = left
- 2 = right

Recorded features:
- dir_up, dir_down, dir_left, dir_right
- food_up, food_down, food_left, food_right
- danger_straight, danger_left, danger_right
- dx_food, dy_food

How to train:
1. After collecting data, run:
   python train_snake_bc.py

Output files:
- snake_smallnet.pth
- snake_biggernet.pth

Model sizes:
- SmallNet:  13 -> 16 -> 3
- BiggerNet: 13 -> 32 -> 16 -> 3

Recommended dataset size:
- minimum: 3000 samples
- better: 5000–10000 samples


Flappy Bird (recording)

Requirements:
- pygame
- numpy

Install (if not installed yet):
1. Open terminal in this folder
2. Run:
   python -m pip install pygame numpy

How to run:
1. Recommended:
   python -m flappy_game
2. Compatibility launcher:
   python flappy_record.py

Controls:
- SPACE or UP: flap
- R: toggle recording on/off
- F: toggle frame saving on/off
- ESC: quit

Recorded data files:
- flappy_game/data/flappy_dataset.csv
- flappy_game/data/states.npy
- flappy_game/data/states_norm.npy
- flappy_game/data/actions.npy
- flappy_game/data/episode_ids.npy
- flappy_game/data/timesteps.npy
- flappy_game/data/dones.npy
- flappy_game/data/frames/frame_XXXXXX.png (when frame saving is ON)

State format (7 floats):
- [bird_x, bird_y, bird_vel, dist_to_pipe_x, pipe_top_y, pipe_bottom_y, pipe_gap_center_y]

How to record Flappy dataset:
1. Run the game:
   python -m flappy_game
2. Press R to start recording.
3. Play the game (SPACE/UP for flap).
4. Press R again to stop and save dataset to files.
5. Optional: press F if you also want PNG frames in frames/.

Flappy dataset fields (name_of_data - описание данных):
- 'bird_x' - 'x-координата птицы в текущем кадре'
- 'bird_y' - 'y-координата птицы в текущем кадре'
- 'bird_vel' - 'вертикальная скорость птицы'
- 'dist_to_pipe_x' - 'расстояние по x до следующей трубы'
- 'pipe_top_y' - 'y верхней границы прохода между трубами'
- 'pipe_bottom_y' - 'y нижней границы прохода между трубами'
- 'pipe_gap_center_y' - 'y центра прохода между трубами'
- 'action' - 'действие игрока: 0 = no flap, 1 = flap'
- 'episode_id' - 'идентификатор эпизода (попытки)'
- 'timestep' - 'номер шага внутри эпизода'
- 'done' - 'флаг конца эпизода: True на шаге столкновения'

Flappy CSV file:
- flappy_game/data/flappy_dataset.csv
- contains one row per game step
- useful for direct training in pandas / sklearn / PyTorch tabular pipeline
- columns include only normalized inputs and the action target

Flappy CSV columns (training only, name_of_data - описание данных):
- 'bird_x_norm' - 'bird_x / SCREEN_W'
- 'bird_y_norm' - 'bird_y / SCREEN_H'
- 'bird_vel_norm' - 'bird_vel / MAX_ABS_VEL (ограничено в диапазоне -1..1)'
- 'dist_to_pipe_x_norm' - 'dist_to_pipe_x / SCREEN_W'
- 'pipe_top_y_norm' - 'pipe_top_y / SCREEN_H'
- 'pipe_bottom_y_norm' - 'pipe_bottom_y / SCREEN_H'
- 'pipe_gap_center_y_norm' - 'pipe_gap_center_y / SCREEN_H'
- 'action' - 'целевая метка: 0 = no flap, 1 = flap'
