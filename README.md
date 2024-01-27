# **ZoeDepth: Combining relative and metric depth** (Official implementation)  <!-- omit in toc -->

Please Refer to the original repo for full details : https://github.com/isl-org/ZoeDepth

### How to run evaluation script:
1. Prepare data in 1 layer directory. For example:
- ground truth in a directory
- inputs in a directory
- output depth estimation in 1 directory

2. run [make_evaluation_text_file.py](https://github.com/clairelin23/ZoeDepth_eval/blob/main/train_test_inputs/make_evaluation_text_file.py) to prepare evaluation txt.file that contains tuples of [input, depth gt, depth estimation]

3. Edit [config.py](https://github.com/clairelin23/ZoeDepth_eval/blob/main/zoedepth/utils/config.py).
   - Create a new config or edit existing config
   - Specify the text file created in step 2 under the "filenames_file_eval" tag.
   - Specify extra parameters such as        
     - "do_kb_crop"
     - "garg_crop"
     - "eigen_crop"
     - "normalize_first" # normalize to relative gt, can be used with "scale_to_absolute" to give better evaluation results, since the goal is to evaluate relative depth performance 
     - "scale_to_absolute" # used for relative gt outputs 
4. run evaluation_pred_read.py
   ```
    python evaluate_pred_ready.py -m {model_name} -d {config_name}
    
    For example:
    python evaluate_pred_ready.py -m zoedepth_nk -d citiscapes_vidar
    ```