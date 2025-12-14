import os

class BaseExperimentLogger:
    def __init__(self, cfg):
        self.config = cfg
        self.logger = None

    def log_config(self, config):
        raise NotImplementedError("Subclasses should implement this method.")

    def log_metric(self, name, value, step=None):
        raise NotImplementedError("Subclasses should implement this method.")

    def end(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def flatten_dict(self, d, parent_key='', sep='.', remove_api_key=True):
        """Flattens a nested dictionary and optionally removes any keys containing 'api_key'."""
        items = []
        for grand_k, grand_v in d.items():
            if isinstance(grand_v, dict):
                for k, v in grand_v.items():
                    if remove_api_key and 'api_key' in k or 'comet' in grand_k or 'wandb' in grand_k:
                        continue

                    new_key = f"{grand_k}{sep}{k}" if grand_k else k
                    if isinstance(v, dict):
                        items.extend(self.flatten_dict(v, new_key, sep=sep, remove_api_key=remove_api_key).items())
                    else:
                        items.append((new_key, v))
            else:
                items.append((grand_k, grand_v))

        return dict(items)

class CometExperimentLogger(BaseExperimentLogger):
    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.get("dry_mode", False):
            print("[MAIN PROCESS] Dry mode enabled: Comet logger not initialized.")
            return
        
        try:
            from comet_ml import Experiment
            
            # Get API key from environment or config
            api_key = os.getenv('COMET_API_KEY')
            if not api_key and 'wandb' in cfg and 'api_key' in cfg['wandb']:
                api_key = cfg['wandb']['api_key']
            
            if not api_key:
                print("Warning: No COMET_API_KEY found in environment or config. Experiment logging disabled.")
                self.logger = None
                return
            
            # Get project and workspace from environment variables or config
            project_name = os.getenv('COMET_PROJECT_NAME', cfg["experiment"]["project"])
            workspace = os.getenv('COMET_WORKSPACE', cfg["experiment"]["entity"])
            
            print(f"[MAIN PROCESS] Initializing Comet experiment with project: {project_name}, workspace: {workspace}")
                
            self.logger = Experiment(
                api_key=api_key,
                project_name=project_name,
                workspace=workspace
            )
            
            # Generate run name based on configuration
            run_name = self._generate_run_name(cfg)
            if run_name:
                self.logger.set_name(run_name)
                print(f"[MAIN PROCESS] Set experiment run name: {run_name}")
            elif "run_name" in cfg.get("experiment", {}):
                self.logger.set_name(cfg["experiment"]["run_name"])
            
            self.log_config(self.config)
            
            # Log relevant code files for this project
            files_to_log = [
                "train.py", "pretrain_model.py", "exp_logger.py", "utils.py",
                "configs/retrieval.yaml", "data/__init__.py",
                "evaluation/evaluate_retrieval.py"
            ]

            for file in files_to_log:
                if os.path.isfile(file):
                    self.logger.log_code(file)
                    print(f"[MAIN PROCESS] Logged code: {file}")
                    
            print("[MAIN PROCESS] Comet experiment logger initialized successfully.")
            
        except ImportError:
            print("Warning: comet_ml not installed. Install with: pip install comet_ml")
            self.logger = None
        except Exception as e:
            print(f"Error initializing Comet logger: {e}")
            self.logger = None

    def _generate_run_name(self, cfg):
        """Generate run name based on configuration parameters"""
        try:
            # Get data field, default to "blip_data" if empty
            data = cfg.get('data', '') or 'blip_data'
            
            # Extract model names (remove path prefixes and special characters)
            siglip_path = cfg.get('siglip_path', 'unknown_siglip')
            siglip_name = siglip_path.split('/')[-1] if '/' in siglip_path else siglip_path
            siglip_name = siglip_name.replace('-', '_')
            
            language_reranker_path = cfg.get('language_reranker_path', 'unknown_language')
            language_name = language_reranker_path.split('/')[-1] if '/' in language_reranker_path else language_reranker_path
            language_name = language_name.replace('-', '_')
            
            # Get learning rate
            init_lr = cfg.get('init_lr', 'unknown_lr')
            
            # Determine training mode
            mode_str = 'finetune' if cfg.get('finetune', False) else 'pretrain'

            # Get current timestamp for uniqueness
            import datetime
            current_date = datetime.datetime.now().strftime("%Y%m%d_%H%M")
            
            # Construct run name
            run_name = f"{data}_{siglip_name}_{language_name}_{init_lr}_{mode_str}_{current_date}"
            
            return run_name
            
        except Exception as e:
            print(f"Error generating run name: {e}")
            return None

    def log_config(self, config):
        if not self.logger:
            return
        flat_config = self.flatten_dict(config)
        flat_config.pop('api_key', None)
        flat_config.pop('wandb', None)
        flat_config.pop('comet', None)
        for k, v in flat_config.items():
            self.logger.log_parameter(k, v)

    def log_metric(self, name, value, step=None):
        if not self.logger:
            return
        try:
            if step is not None:
                self.logger.log_metric(name, value, step=step)
            else:
                self.logger.log_metric(name, value)
        except Exception as e:
            print(f"Error logging metric {name}: {e}")

    def end(self):
        if not self.logger:
            return
        try:
            self.logger.end()
        except Exception as e:
            print(f"Error ending experiment: {e}")

    def log_checkpoint(self, path, epoch=None):
        """Log a saved checkpoint path to Comet.

        Parameters
        ----------
        path : str
            Filesystem path to the saved checkpoint file.
        epoch : int, optional
            Epoch number associated with this checkpoint; used as step.
        """
        if not self.logger:
            return

        normalized_path = os.path.abspath(path)
        # Store per-epoch path (if provided) and always update latest
        if epoch is not None:
            self.logger.log_parameter(f"checkpoint.{epoch}.path", normalized_path)
            self.logger.log_parameter("checkpoint.latest.path", normalized_path)


class ExperimentLogger:
    def __new__(cls, cfg):
        if cfg.get("dry_mode", False):
            print("[MAIN PROCESS] Dry mode enabled: No logger will be used.")
            return None
        backend = cfg.get("experiment", {}).get("backend", "wandb").lower()
        if backend == "comet":
            print(f"[MAIN PROCESS] Initializing {backend} experiment logger...")
            return CometExperimentLogger(cfg)
        else:
            print(f"[MAIN PROCESS] Backend '{backend}' not recognized; logger disabled.")
            return None