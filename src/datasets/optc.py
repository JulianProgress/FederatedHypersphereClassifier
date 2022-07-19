import json
import os
from pathlib import Path

import gzip
import pandas as pd
import tqdm


class OPTCDatasetGenerator(object):

    CATEGORY_COLUMN_NAME_LIST = ["acuity_level"]
    NUMERIC_COLUMN_NAME_LIST = ["size"]

    def __init__(self):
        super(OPTCDatasetGenerator).__init__()
        pass

    @classmethod
    def generate_df(cls, file_path):
        metadata_lines = list()
        properties_lines = list()
        error_count = 0

        with gzip.open(file_path, 'r') as fin:
            for i, line in enumerate(fin):
                try:
                    json_line = line.decode('utf8').replace("'", '"')
                    data = json.loads(json_line)

                    properties = data['properties']
                    del data['properties']
                    metadata = data

                    # Pop for metadata
                    for column_name in OPTCDatasetGenerator.CATEGORY_COLUMN_NAME_LIST:
                        if column_name in properties:
                            metadata[column_name] = properties.pop(column_name)
                    for column_name in OPTCDatasetGenerator.NUMERIC_COLUMN_NAME_LIST:
                        if column_name in properties:
                            metadata[column_name] = properties.pop(column_name)
                    if "end_time" in properties and "start_time" in properties:
                        metadata["elapsed_time"] = int(properties.pop("end_time")) - int(properties.pop("start_time"))
                except Exception as e:
                    error_count += 1

                properties_lines.append(properties)
                metadata_lines.append(metadata)

                if (i + 1) % 100000 == 0:
                    print('%d th line done' % (i + 1))

                if (i + 1) > 30000000:
                    break

        metadata_df = pd.DataFrame(metadata_lines)
        properties_df = pd.DataFrame(properties_lines)

        # Fillna
        metadata_df["size"] = metadata_df["size"].fillna(0)
        metadata_df["elapsed_time"] = metadata_df["elapsed_time"].fillna(0)

        properties_df = properties_df.fillna("none")

        return dict(
            metadata_df=metadata_df,
            properties_df=properties_df
        )

    @classmethod
    def save_by_host(cls, save_dir_path, metadata_df, properties_df, file_idx=0):
        host_idx_list_dict = {host_name: metadata_df.index[metadata_df["hostname"] == host_name] for host_name in set(metadata_df["hostname"])}

        # Generate save directory
        Path(save_dir_path).mkdir(exist_ok=True)

        for host_name, idx_list in host_idx_list_dict.items():
            host_name_for_save = host_name.replace(".", "-")

            properties_data_save_file_path = os.path.join(save_dir_path, f"{host_name_for_save}_properties_{file_idx}.csv")
            metadata_data_save_file_path = os.path.join(save_dir_path, f"{host_name_for_save}_metadata_{file_idx}.csv")

            properties_host_df = properties_df.iloc[idx_list]
            properties_host_df.to_csv(properties_data_save_file_path, index=False)

            metadata_host_df = metadata_df.iloc[idx_list]
            metadata_host_df.to_csv(metadata_data_save_file_path, index=False)

            print(f"Complete to save : {host_name_for_save}-{file_idx}")

    @classmethod
    def generate_host_df_all(cls, file_path_list, save_dir_path=None):
        for file_idx, file_path in enumerate(tqdm.tqdm(file_path_list)):
            generated_df_dict = OPTCDatasetGenerator.generate_df(file_path=file_path)

            if save_dir_path:
                OPTCDatasetGenerator.save_by_host(
                    save_dir_path=save_dir_path,
                    metadata_df=generated_df_dict["metadata_df"],
                    properties_df=generated_df_dict["properties_df"],
                    file_idx=file_idx
                )
