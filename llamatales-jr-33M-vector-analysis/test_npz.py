import sys
sys.path.append("/home/a3murali/clustering/installs")

import numpy as np
import pandas as pd

loaded_data = np.load('story_dataset_10.npz', allow_pickle = True)
np_array = loaded_data['array']

# Convert back to DataFrame if needed
df = pd.DataFrame(np_array, columns = ["prompt", "story", "hidden_states", "output_token_prompt_id"])
print(df)