import wandb

class Agent():
    
    def __init__(self, project, config, model_fn, train_fn):
        self.project = project
        self.sweep_id = wandb.sweep(config)
        self.model_fn = model_fn
        self.train_fn = train_fn
        
    def sweep(self, count):
        def train():
            return self._train(count)
        
        wandb.agent(self.sweep_id, function=train, count=count, project=self.project)
    
    def _train(self, count):
        with wandb.init() as run:
            model = self.model_fn(run.config)
            
            hist = {
                'train_losses': [],
                'test_losses': [],
                'train_accuracies': [],
                'test_accuracies': []
            }

            self.train_fn(model, run.config, hist, wandb_active=True)
            
            return hist