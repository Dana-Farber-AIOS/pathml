import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
import sys
import copy
import itertools
import pandas as pd
import sklearn
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import itertools
from io import StringIO
import requests
import json
import numpy as np
import subprocess

class TCGA():

    def __init__ (self, tissue_codes, outdir):
        print("Creating a TCGA class")
        self.tissue_codes = tissue_codes
        self.outdir = outdir
        self.fetch_data(tissue_codes, outdir)
    
    def fetch_data(self, tissue_codes, outdir):
        fields = [
            "file_name",
            "cases.submitter_id",
            "cases.samples.sample_type",
            "cases.disease_type",
            "cases.project.project_id",
            "cases.annotations.category",
            "experimental_strategy",
            "cases.annotations.entity_type",
            "data_type",
            "file_id"
        ]

        fields = ",".join(fields)

        cases_endpt = "https://api.gdc.cancer.gov/cases"
        files_endpt = "https://api.gdc.cancer.gov/files"

        # This set of filters is nested under an 'and' operator - these narrow down the set of data to be downloaded.
        filters = {
            "op": "and",
            "content":[
                {
                "op": "in",
                "content":{
                    "field": "files.cases.project.project_id",
                    #the value here is a list which holds the names of cohorts to be downloaded - this is the most likely to be changed.
                    "value": tissue_codes 
                    }
                },
                {
                "op": "in",
                "content":{
                    "field": "files.cases.samples.sample_type",
                    "value": ["Primary Tumor"] #TODO look into other tumors except primary
                    }
                },
                {
                "op": "in",
                "content":{
                    "field": "files.data_type",
                    #Can retrieve other data types as necessary by adding keyword strings to this list.
                    "value": ["Slide Image"]
                    }
                },
                {
                "op": "in",
                "content":{
                    "field": "files.experimental_strategy",
                    #NOTE - you may want to remove this entry of the dictionary when trying to download normal tissue, as these slides may not be labeled as "diagnostic"
                    "value": ["Diagnostic Slide"]
                    }
                },
            ]
        }

        # A POST is used, so the filter parameters can be passed directly as a Dict object.
        params = {
            "filters": filters,
            "fields": fields,
            "format": "json",
            "size": "2000"
            }

        # The parameters are passed to 'json' rather than 'params' in this case
        response = requests.post(files_endpt, headers = {"Content-Type": "application/json"}, json = params)
        # pd.read_table(StringIO(response.text), low_memory=False).head()

        # response.text
        tumor_wsi_hits = response.json()['data']['hits']
        #Check the size of the dataset
        print(len(tumor_wsi_hits), 'total samples found')

        #create the df with two columns: one containing the file ID, and the other containing the name of the WSI file.
        df_tumor_TGCT = pd.DataFrame({"uuid": [x['file_id'] for x in tumor_wsi_hits if "DX1" in x["file_name"]],"filename": [x['file_name'] for x in tumor_wsi_hits if "DX1" in x["file_name"]]})

        #now, a new column needs to be built and added to the df - this column will contain the paths from which the WSI file itself will be fetched.
        isb_root = 'gs://gdc-tcga-phs000178-open'
        full_paths = []
        for x in tumor_wsi_hits:
            if "DX1" in x["file_name"]:
                uuid = x['file_id']
                filename = x['file_name']
                full_path = "/".join([isb_root, uuid,filename])
                full_paths.append(full_path)
        
        df_tumor_TGCT['full_gs_path'] = full_paths

        #display a random selection of 5 paths - a good sanity check
        print(df_tumor_TGCT['full_gs_path'].sample(5).values)

        """Oftentimes, the dataset of interest can contain hundreds of WSIs, which makes for an extremely bulky download. It's therefore good practice to split the downloading process into small batches.

        To do so, the paths are extracted from the database, split, and then written into a shell script, which is run directly from the command line."""


        #Extract the paths of interest, and split them into a smaller number of batches.
        num_splits = int(len(df_tumor_TGCT['full_gs_path']) / 50)
        splits = np.array_split(df_tumor_TGCT['full_gs_path'].values, num_splits)

        #The goal is to have at most around 50 paths per batch; more than this could cause a job to crash or be terminated. Check the shapes and change the number of splits as necessary.
        print('Downloading in batches of', [x.shape for x in splits])

        #choose a name for the download file
        filename = "TCGA_dl.sh"
        #.sh extension = bash script

        #Write into the chosen file
        with open(filename,'w') as file:
            file.write('#!/bin/bash/\n')


            split_counter = 0
            for paths in splits:
                tempJoin = ' '.join(paths)
                tempCmd = 'gsutil -m cp -L split_{}.log '.format(split_counter)
                #{} changes what's in string
                tempCmd += tempJoin
                tempCmd += ' '
                tempCmd += outdir
                #going to download to current directory

                split_counter += 1

                os.system(tempCmd)

                file.write(tempCmd+'\n\n')

"""
By this point, you will have written the .sh file which, when run, will execute the download of your chosen dataset in 50-slide batches. This process usually takes a couple of hours.

Because the shell script downloads to the current directory, the script should be moved to the appropriate folder in a persistent disk before it is run; otherwise, it could start downloading to a small boot disk and cause memory issues.
"""


