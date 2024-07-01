Sample repo for fine-tuning a custom multi-head pytorch model based on a BERT encoder model. 

Code likely wont work right out of the box, rather this repo serves as a simple template to get started. 

Data is currently expected to be a json-line (.jsonl) file with a list of dicts as the inputs. 

Likely code changes to `encoder_model/data.py`, `encoder_model/model.py`, and `./train.py` files to be successful for your use case. Additionally, only simple evals are done currently. Likely would want to build your own custom validation handler class/file. 
