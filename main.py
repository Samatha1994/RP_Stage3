
# import os
# import shutil
# import sys
# import json
# from keyword_extractor import extract_keywords
# from pygoogle_image import image as pi
# from image_classification import classify_images_for_solution
# import model as md
# from image_processor import process_and_classify_images

# def main():
#     # source_folder = "/homes/samatha94/ExAI/outputs/config_files"
#     # destination_base_folder = "/homes/samatha94/ExAI/outputs/Google_images"
#     # model_path = "/homes/samatha94/ExAI/outputs/model_resnet50V2_10classes_retest2023June.h5"
#     # base_output_path = "/homes/samatha94/ExAI/outputs"

#     # "source_folder": "C:\\Users\\Dell-PC\\OneDrive - Kansas State University\\Desktop\\DataSemantics prep\\ExAI_2\\outputs\\config_files",
#     # "destination_base_folder": "C:\\Users\\Dell-PC\\OneDrive - Kansas State University\\Desktop\\DataSemantics prep\\ExAI_2\\outputs\\Google_images",
#     # "model_path": "C:\\Users\\Dell-PC\\OneDrive - Kansas State University\\Desktop\\DataSemantics prep\\ExAI_2\\outputs\\model_resnet50V2_10classes_retest2023June.h5",
#     # "base_output_path": "C:\\Users\\Dell-PC\\OneDrive - Kansas State University\\Desktop\\DataSemantics prep\\ExAI_2\\outputs"
    
#     with open('config.json', 'r') as config_file:
#         config = json.load(config_file)

    

#     destination_base_folder = config["destination_base_folder"]
#     test_directory = destination_base_folder
#     model_path = config["model_path"]
#     base_output_path = config["base_output_path"]    
  
#     model, layer_outputs, layer_names, feature_map_model = md.load_and_analyze_model(model_path)   
   
#     # process_and_classify_images(feature_map_model, test_directory,base_output_path)

#     for class_name in os.listdir(destination_base_folder):
#         class_path = os.path.join(destination_base_folder, class_name)
        
#         # Check if the path is indeed a directory
#         if os.path.isdir(class_path):
#             # Process and classify images for each class
#             process_and_classify_images(feature_map_model, class_path, base_output_path)

# if __name__ == "__main__":
#     main()

import os
import json
import pandas as pd
import model as md
from image_processor import process_and_classify_images

def main():
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)

#         {
    
#     "destination_base_folder": "C:\\Users\\Dell-PC\\OneDrive - Kansas State University\\Desktop\\DataSemantics prep\\for rushrukh project\\ExAI_2\\outputs\\validation",
#     "model_path": "C:\\Users\\Dell-PC\\OneDrive - Kansas State University\\Desktop\\DataSemantics prep\\for rushrukh project\\ExAI_2\\outputs\\model_resnet50V2_10classes_retest2023June.h5",
#     "base_output_path": "C:\\Users\\Dell-PC\\OneDrive - Kansas State University\\Desktop\\DataSemantics prep\\for rushrukh project\\ExAI_2\\outputs"    
# }

    destination_base_folder = config["destination_base_folder"]
    model_path = config["model_path"]
    base_output_path = config["base_output_path"]    
    model, layer_outputs, layer_names, feature_map_model = md.load_and_analyze_model(model_path)   

    all_train_dfs = []
    all_val_dfs = []

    # Create the evaluation and verification directories
    eval_dir = os.path.join(base_output_path, 'evaluation')
    verif_dir = os.path.join(base_output_path, 'verification')
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(verif_dir, exist_ok=True)

    for class_name in os.listdir(destination_base_folder):
        class_path = os.path.join(destination_base_folder, class_name)
        if os.path.isdir(class_path):
            train_df, val_df = process_and_classify_images(feature_map_model, class_path, base_output_path)
            all_train_dfs.append(train_df)
            all_val_dfs.append(val_df)

    # Concatenate all DataFrames and save to CSV
    final_train_df = pd.concat(all_train_dfs, ignore_index=True)
    final_val_df = pd.concat(all_val_dfs, ignore_index=True)
    final_train_df.to_csv(os.path.join(eval_dir, 'evaluation_set.csv'), index=False)
    final_val_df.to_csv(os.path.join(verif_dir, 'verification_set.csv'), index=False)

if __name__ == "__main__":
    main()

