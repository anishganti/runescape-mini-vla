from src.scripts.load_data import get_episodes, load_actions
import os

base_dir = "/Users/anishganti/runescape_mini_vla/data/mining"
def total_actions():
    total_press, total_release, total_wait = 0, 0, 0
    episodes = get_episodes(base_dir)
    for episode in episodes:
        actions = load_actions(episode)
        press_count, release_count, wait_count = get_action_counts(actions)
        total_press += press_count
        total_release += release_count
        total_wait += wait_count
        print(f"Episode {episode}: Press {press_count}, Release {release_count}, Wait {wait_count}")
    print(f"Total Press: {total_press}, Total Release: {total_release}, Total Wait: {total_wait}")
    return total_press, total_release, total_wait

def get_action_counts(actions):
    press_count = 0
    release_count = 0
    wait_count = 0
    for action in actions:
        if action["a"] == 0:
            press_count += 1
        elif action["a"] == 1:
            release_count += 1
        elif action["a"] == 2:
            wait_count += 1
    return press_count, release_count, wait_count

total_actions()