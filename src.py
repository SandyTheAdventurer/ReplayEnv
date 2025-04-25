import time
from absl import app
from pysc2 import run_configs
from pysc2.lib.replay import get_replay_version
from s2clientprotocol import sc2api_pb2 as sc_pb

#Object recieves a valid path for StarCraft II replay file
# and starts a replay game
# with the given parameters.
class ReplayEnv():
    def __init__(
            self,
            replay,
            observed_player,
            window_size=(640, 480),
            feature_screen_size=(84, 84),
            feature_minimap_size=(64, 64),
            full_screen=False,
            fps=22.4,
            step_mul=1,
            max_game_steps=0,
            feature_camera_width=24,
        ):
        run_config = run_configs.get()
        interface = sc_pb.InterfaceOptions()
        self.window_size = window_size
        self.full_screen = full_screen
        self.step_mul = step_mul
        self.fps = fps
        self.max_game_steps = max_game_steps
        interface.raw = True
        interface.raw_affects_selection = True
        interface.raw_crop_to_playable_area = True
        interface.score = True
        interface.show_cloaked = True
        interface.show_burrowed_shadows = True
        interface.show_placeholders = True
        interface.feature_layer.width = feature_camera_width
        interface.feature_layer.resolution.x,interface.feature_layer.resolution.y  = feature_screen_size
        interface.feature_layer.minimap_resolution.x, interface.feature_layer.minimap_resolution.y = feature_minimap_size
        interface.feature_layer.crop_to_playable_area = True
        interface.feature_layer.allow_cheating_layers = True
        replay_data = run_config.replay_data(replay)
        self.start_replay = sc_pb.RequestStartReplay(
            replay_data=replay_data,
            options=interface,
            disable_fog=False,
            observed_player_id=observed_player)
        version = get_replay_version(replay_data)
        self.run_config = run_configs.get(version=version)
    
        self.controller_context = self.run_config.start(
            full_screen=self.full_screen,
            window_size = self.window_size,
        )
        self.controller = self.controller_context.__enter__()
        info = self.controller.replay_info(replay_data)
        print(" Replay info ".center(60, "-"))
        print(info)
        print("-" * 60)
        map_path = info.local_map_path
        if map_path:
            self.start_replay.map_data = self.run_config.map_data(map_path, len(info.player_info))
        self.controller.start_replay(self.start_replay)
    
    def step(self):
        frame_start_time = time.time()
        self.controller.step(self.step_mul)
        obs = self.controller.observe()
        time.sleep(max(0, frame_start_time + 1 / self.fps - time.time()))
        return obs
    
    def close(self):
        self.controller.close()
        self.controller_context.__exit__(None, None, None)

def main(unused_argv):
    replay = "1.SC2Replay"
    env = ReplayEnv(replay, 2, fps=60)
    obs = env.step()
    while not obs.player_result:
        obs = env.step()
        print("Step: ", obs.observation.game_loop)
    print("Score: ", obs.observation.score.score)
    print("Result: ", obs.player_result)
    
if __name__ == "__main__":
    app.run(main)