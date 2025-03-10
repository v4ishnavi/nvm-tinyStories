# nvm-tinyStories

the following is the structure of the folders. 
```
-artifacts 
-- transformer-iterations
--- config.toml
--- model.pt
-- tokenizer.json

-evaluation
--calc_params.py
--combined_processes.sh
--plot_losses.py
--plot.ipynb
--scoring_linear.py
--scoring.py

-LinearLMmain
--neptune
--artifacts/linear_decoder 
---checkpoints 
--configs
---config_linear.json
--- (other configs)

--models
---__init__.py
--- linear_model.py

--torch_datasets
---product.py
--gitignore
--linear_decoder.py
--readme

-runs
--all the toml files for running

-scripts
--ada-generation-run.sh
--ada-profile.sh
--ada-sync.sh
--ada-training-run.sh

-reports
--interim.pdf
--final.pdf


-gitignore
-config.py
-dataset.py
-mechanterp_test.py
-models.py
-Readme
-requirements.txt
-test.py
-train.py
```
the github link is : https://github.com/v4ishnavi/nvm-tinyStories
