import os
import csv


class Recorder:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.csv_path = os.path.join(data_dir, "flappy_dataset.csv")
        self.csv_fieldnames = [
            "bird_x_norm",
            "bird_y_norm",
            "bird_vel_norm",
            "dist_to_pipe_x_norm",
            "pipe_top_y_norm",
            "pipe_bottom_y_norm",
            "pipe_gap_center_y_norm",
            "action",
        ]
        self.csv_rows_buffer = []
        self.global_frame_id = 0

    def record_frame(self, state, state_norm, action, episode_id, timestep, done):
        self.csv_rows_buffer.append(
            {
                "bird_x_norm": state_norm[0],
                "bird_y_norm": state_norm[1],
                "bird_vel_norm": state_norm[2],
                "dist_to_pipe_x_norm": state_norm[3],
                "pipe_top_y_norm": state_norm[4],
                "pipe_bottom_y_norm": state_norm[5],
                "pipe_gap_center_y_norm": state_norm[6],
                "action": action,
            }
        )
        self.global_frame_id += 1

    def has_pending_rows(self):
        return bool(self.csv_rows_buffer)

    def flush(self):
        if not self.csv_rows_buffer:
            return 0

        write_header = not os.path.exists(self.csv_path)

        with open(self.csv_path, "a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.csv_fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(self.csv_rows_buffer)

        saved_count = len(self.csv_rows_buffer)
        self.csv_rows_buffer.clear()
        return saved_count

    def undo_last(self):
        if self.csv_rows_buffer:
            self.csv_rows_buffer.pop()
            self.global_frame_id = max(0, self.global_frame_id - 1)
            return "buffer"

        if not os.path.exists(self.csv_path):
            return None

        with open(self.csv_path, "r", newline="", encoding="utf-8") as csv_file:
            rows = list(csv.DictReader(csv_file))

        if not rows:
            return None

        rows.pop()
        with open(self.csv_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.csv_fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        return "saved"
